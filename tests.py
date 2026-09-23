"""Self-contained tests. Run: ``python tests.py`` (no data or pytest needed).

Each check is a function that raises on failure; ``main`` runs them all and
reports. The homography round-trip matters most: it is the only test that can
catch a silent coordinate-frame error, which would quietly corrupt every
spatial feature.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from zfsa import bouts, config, features, geometry, gmm, metadata, naming, normalize  # noqa: E402


def _homography(scale=3.0, tx=120.0, ty=45.0, rot=0.21):
    """A similarity transform written as a homography, plus mild perspective."""
    c, s = np.cos(rot), np.sin(rot)
    return np.array([[scale * c, -scale * s, tx],
                     [scale * s, scale * c, ty],
                     [1e-4, 5e-5, 1.0]])


def _apply(H, pts):
    p = np.hstack([pts, np.ones((len(pts), 1))])
    q = p @ H.T
    return q[:, :2] / q[:, 2:3]


def test_homography_roundtrip():
    """Five known fish, pushed to pixels and back through Calibration, keep their distances."""
    canon = np.array([[0.20, 0.20], [0.80, 0.25], [0.50, 0.50], [0.30, 0.75], [0.70, 0.80]])
    H = _homography()
    pixel = _apply(H, canon)
    cal = geometry.Calibration("r", np.linalg.inv(H), 6, 0.0, 0.0, [], True)
    back = cal.to_canonical(pixel.astype(np.float32))
    assert np.abs(back - canon).max() < 1e-4, "round trip through to_canonical failed"

    platform = np.array([0.20, 0.20])
    d_canon = np.linalg.norm(back - platform, axis=1)
    expected = np.array([0.0, 0.60207973, 0.42426407, 0.55901699, 0.78102497])
    assert np.allclose(d_canon, expected, atol=1e-4), d_canon
    d_pixel = np.linalg.norm(pixel - _apply(H, platform[None])[0], axis=1)
    assert not np.allclose(d_canon, d_pixel), "pixel and canonical distances identical"


def test_bout_collapsing():
    """Known trigger series collapses to the bout structure it was built from."""
    t = [11.0, 11.6, 12.2, 12.8, 60.0, 200.0, 202.0]  # a 4-trigger loiter, a single, a pair
    ev = pd.DataFrame({"run_id": "r", "platform": "active", "t_seconds": t, "is_artifact": False})
    B = bouts.build_bout_table(ev, gaps_s=(5.0,), window=(10.0, 2900.0), bin_s=600.0)
    assert len(B) == 3, f"expected 3 bouts, got {len(B)}"
    first = B.sort_values("t_start").iloc[0]
    assert first.n_triggers == 4
    assert abs(first.duration_s - 1.8) < 1e-9
    assert first.n_at_floor == 3, f"floor intervals {first.n_at_floor}"
    assert B.in_window.all() and B.overlaps_window.all()
    assert len(bouts.build_bout_table(ev, gaps_s=(50.0,), window=(10.0, 2900.0))) == 2
    assert len(bouts.build_bout_table(ev, gaps_s=(0.5,), window=(10.0, 2900.0))) == 7


def test_window_flags_differ_only_at_the_boundary():
    """A bout straddling the window end is counted but not measured."""
    ev = pd.DataFrame({"run_id": "r", "platform": "active",
                       "t_seconds": [100.0, 2899.0, 2901.0], "is_artifact": False})
    B = bouts.build_bout_table(ev, gaps_s=(5.0,), window=(10.0, 2900.0))
    straddle = B[B.t_start == 2899.0].iloc[0]
    assert straddle.overlaps_window and not straddle.in_window


def test_exposure_offset():
    """Observed time is wall time minus blind time, resolved per bin."""
    ev = pd.DataFrame({"run_id": "r", "platform": "active",
                       "t_seconds": [100.0, 200.0, 700.0], "is_artifact": False})
    runs = pd.DataFrame({"run_id": ["r"], "actual_duration_s": [3000.0]})
    X = bouts.build_exposure_table(ev, runs, blind_s_per_trigger=0.2, window=(10.0, 2900.0), bin_s=600.0)
    b0 = X[X.time_bin == 0].iloc[0]
    assert abs(b0.wall_s - 590.0) < 1e-9, "bin 0 starts at the window, not at 0"
    assert b0.n_triggers_in_bin == 2
    assert abs(b0.observed_time_s - (590.0 - 0.4)) < 1e-9
    assert abs(X[X.time_bin == 4].iloc[0].wall_s - 500.0) < 1e-9, "bin 4 is truncated by the window end"


def test_effort_subsample_is_without_replacement():
    B = pd.DataFrame({"run_id": ["r"] * 30, "gap_s": 5.0, "platform": "active",
                      "bout_id": range(30), "in_window": True})
    idx, status = bouts.effort_subsample_index(B, k=20, b=50, seed=7)
    assert status.effort_equalized.all()
    per = idx.groupby("draw").bout_id.agg(["size", "nunique"])
    assert (per["size"] == 20).all() and (per["nunique"] == 20).all(), "drew a duplicate"
    idx2, _ = bouts.effort_subsample_index(B, k=20, b=50, seed=7)
    assert idx.equals(idx2), "not reproducible from the seed"
    _, st = bouts.effort_subsample_index(B.iloc[:8].copy(), k=20, b=50, seed=7)
    assert not st.effort_equalized.any(), "a run below k must not be rescued"


def test_eb_centering_tracks_the_day_effect():
    """Shrinkage responds to how big the day effect actually is."""
    rng = np.random.default_rng(1)
    dates = pd.Series(np.repeat([f"d{i}" for i in range(20)], 2))
    veh = pd.Series(True, index=range(40))
    _, flat = normalize.eb_day_center(pd.Series(rng.normal(0, 1, 40)), dates, veh)
    day = pd.Series(np.repeat(rng.normal(0, 5, 20), 2) + rng.normal(0, 0.2, 40))
    _, strong = normalize.eb_day_center(day, dates, veh)
    assert strong["weight"] > 0.9 and flat["weight"] < 0.5, (strong["weight"], flat["weight"])


def test_guarded_scaling_survives_a_mass_point():
    """A feature whose controls are mostly identical is scaled by the SD, never by ~0."""
    runs = pd.DataFrame({"run_id": [f"r{i}" for i in range(12)], "date": ["d"] * 12,
                         "is_vehicle": [True] * 10 + [False] * 2})
    F = pd.DataFrame({"run_id": runs.run_id, "massy": [0.0] * 8 + [0.1, 0.2, 5.0, 9.0],
                      "normal": np.linspace(0, 10, 12)})
    Z, rep, _ = normalize.scale_guarded(F, runs, {})
    assert np.isfinite(Z[["massy", "normal"]].to_numpy()).all()
    rep = rep.set_index("feature")
    assert rep.loc["massy", "method"].startswith("dmso_sd") and rep.loc["normal", "method"] == "mad"


def test_permutation_preserves_the_day_structure():
    cond = np.array(["DMSO", "X", "Y", "DMSO", "X", "Z", "DMSO", "Y", "Z"])
    date = np.array(["d1"] * 3 + ["d2"] * 3 + ["d3"] * 3)
    rng = np.random.default_rng(3)
    for _ in range(50):
        perm = gmm.permute_within(cond, date, rng)
        for d in np.unique(date):
            assert sorted(cond[date == d]) == sorted(perm[date == d]), "label set changed within a day"


def test_rim_band():
    got = features.in_rim_band(np.array([0.5, 0.02, 0.5, 0.98]), np.array([0.5, 0.5, 0.03, 0.5]))
    assert got.tolist() == [False, True, True, True]


def test_naming():
    aliases = config.DEFAULTS["metadata"]["compound_aliases"]
    assert naming.parse_group("20260331_140838_self_admin_g3_5_fish_DMSO_control") == "3", "g3_5_fish is group 3"
    assert naming.parse_group("20260101_120000_self_admin_g2.5_DMSO") == "2.5"
    assert naming.parse_group("20260101_120000_self_admin_g2_5_ketamine") == "2.5"
    assert naming.parse_group("20260101_120000_self_admin_DMSO") is None
    d, _ = naming.parse_dose("20260303_175435_self_admin_g1_ketamine_20mg per L")
    assert (d.value, d.unit) == (20.0, "mg/L")
    d, _ = naming.parse_dose("x_g1_MK801_20uM")
    assert (d.value, d.unit) == (20.0, "uM")
    assert naming.parse_compound("20260303_155417_self_admin_g1_DMSO", aliases) == ("DMSO", True)
    assert naming.parse_compound("20260304_152759_self_admin_g2_h2CBD_natural_10uM", aliases) == ("H2CBD_NAT", False)
    assert naming.parse_compound("20260101_120000_self_admin_g1_9533_3uM", aliases) == ("9533", False)
    assert naming.parse_failure("20260612_152912_self_admin_g1_methadone_25uM_bugged_pump") == "bugged pump"
    assert naming.parse_trigger_image("active_trigger_0001_frame_000002_20260303_155417.jpg").frame_idx == 2


def _fake_runs(root: Path, names: list[str]) -> None:
    for n in names:
        d = root / n
        (d / "images").mkdir(parents=True)
        for f in ("active_events.tsv", "inactive_events.tsv"):
            (d / f).write_text("frame_idx\tt_seconds\tsignal\tbaseline\n", encoding="utf-8")
        (d / "params.json").write_text(json.dumps({}), encoding="utf-8")


def test_metadata_draft_and_validation():
    names = ["20260101_100000_self_admin_g1_DMSO", "20260101_110000_self_admin_g1_MK801_20uM",
             "20260102_100000_self_admin_g2_DMSO", "20260102_110000_self_admin_DMSO",
             "20260102_120000_self_admin_g2_9533_3uM_bugged_pump"]
    cfg = config.DEFAULTS["metadata"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _fake_runs(root, names)
        table, issues = metadata.draft(root, aliases=cfg["compound_aliases"], positive_controls=cfg["positive_controls"])
        assert len(table) == 5 and (table.is_vehicle == "true").sum() == 3
        assert table.set_index("run_id").loc["20260101_110000", "is_positive_control"] == "true"
        assert table.set_index("run_id").loc["20260102_120000", "flag"] == "bugged pump"
        assert any("no fish group" in i for i in issues) and any("bugged pump" in i for i in issues)
        csv = root / "metadata.csv"
        table.to_csv(csv, index=False)
        runs, dirs, _ = metadata.load(csv, root)
        assert len(runs) == 5 and set(dirs) == set(runs.run_id)
        assert runs.set_index("run_id").loc["20260102_110000", "run_order_within_day"] == 2
        R = normalize.resolve_group(runs)
        assert R.set_index("run_id").loc["20260102_110000", "fish_group_resolved"] == "2", "same-day group fill"

        # a run folder with no row, a bad boolean, and too few vehicles must all be refused
        for mutate, msg in [
            (lambda t: t.iloc[1:], "no row"),
            (lambda t: t.assign(exclude=["maybe"] + ["false"] * 4), "not true/false"),
            (lambda t: t.assign(exclude=["true", "false", "true", "false", "false"]), "vehicle run"),
        ]:
            mutate(table).to_csv(csv, index=False)
            try:
                metadata.load(csv, root)
            except metadata.MetadataError as exc:
                assert msg in str(exc), f"wrong error for {msg!r}: {exc}"
            else:
                raise AssertionError(f"metadata with {msg!r} was accepted")


def test_gmm_finds_planted_structure_and_not_absent_structure():
    a = config.DEFAULTS["analysis"]
    grid_kw = {"k_max": a["k_max"], "cov_types": a["covariance_types"], "n_init": 5,
               "min_cluster_n": a["min_cluster_n"]}
    res = gmm.self_test(a["seed"], grid_kw, a["tie_dbic"])
    assert res["planted_K"] == 2 and res["single_gaussian_K1_rate"] >= 0.9, res


TESTS = [v for k, v in globals().items() if k.startswith("test_")]


def main() -> int:
    failed = []
    for fn in TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed.append(fn.__name__)
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - len(failed)}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
