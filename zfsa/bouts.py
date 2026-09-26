"""Bout-level and exposure tables, and the effort-equalisation draws.

**Bouts are recomputed from the event times**, at the configured bout gap
(``events.bout_gap_s``), rather than read from the event table.

**Two window flags, because they answer different questions.** A bout counts
toward the in-window *count* if any of its triggers falls inside the window;
that is ``overlaps_window``. It is the wrong rule for a *duration* -- a bout
straddling the boundary has a truncated duration that is an artifact of where
the window was cut -- so ``in_window`` is the strict flag, and it is what the
duration features and the pose features use.

Because the window trims a contiguous prefix and suffix, it cannot merge or
split interior bouts, so ``overlaps_window`` counts must equal the per-run
in-window bout counts from :func:`zfsa.events.summarise_run`. The pipeline
asserts that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import events as ev_mod

#: Prespecified bout gap. Fixed before outcomes were examined.
PRIMARY_GAP_S = 5.0
BOUT_GAPS_S: tuple[float, ...] = (PRIMARY_GAP_S,)

#: Hardware floor between successive triggers.
REFRACTORY_S = 0.6

#: An inter-event interval within ``REFRACTORY_S + FLOOR_TOL_S`` is "at the
#: floor": the fish is parked on the platform and the pump, not the fish, is
#: setting the rate.
FLOOR_TOL_S = 0.1

BOUT_COLUMNS = [
    "run_id",
    "gap_s",
    "platform",
    "bout_id",
    "t_start",
    "t_end",
    "duration_s",
    "n_triggers",
    "n_at_floor",
    "n_triggers_in_window",
    "time_bin",
    "in_window",
    "overlaps_window",
]


def _bouts_for_series(
    t: pd.Series,
    *,
    gap_s: float,
    window: tuple[float, float],
    bin_s: float,
    floor_s: float,
) -> pd.DataFrame:
    """Collapse one run/platform's post-artifact trigger times into bouts."""
    lo, hi = window
    bid = ev_mod.assign_bouts(t, gap_s)
    df = pd.DataFrame({"t": t.to_numpy(), "bout_id": bid.to_numpy()})
    df["in_win"] = (df.t >= lo) & (df.t <= hi)
    # Intervals at the pump floor, counted within each bout.
    df["at_floor"] = (df.t.diff() <= floor_s) & (df.bout_id == df.bout_id.shift())

    out = df.groupby("bout_id", sort=True).agg(
        t_start=("t", "min"),
        t_end=("t", "max"),
        n_triggers=("t", "size"),
        n_triggers_in_window=("in_win", "sum"),
        n_at_floor=("at_floor", "sum"),
    ).reset_index()
    out["duration_s"] = out.t_end - out.t_start
    out["time_bin"] = ev_mod.time_bin(out.t_start, bin_s)
    out["in_window"] = (out.t_start >= lo) & (out.t_end <= hi)
    out["overlaps_window"] = out.n_triggers_in_window > 0
    return out


def build_bout_table(
    events: pd.DataFrame,
    *,
    gaps_s: tuple[float, ...] | list[float] = BOUT_GAPS_S,
    window: tuple[float, float] = (10.0, 2900.0),
    bin_s: float = 600.0,
    refractory_s: float = REFRACTORY_S,
    floor_tol_s: float = FLOOR_TOL_S,
) -> pd.DataFrame:
    """One row per bout, per gap, per platform, across all runs."""
    floor_s = refractory_s + floor_tol_s
    post = events.loc[~events.is_artifact, ["run_id", "platform", "t_seconds"]]
    post = post.sort_values(["run_id", "platform", "t_seconds"], kind="stable")

    frames: list[pd.DataFrame] = []
    for gap in gaps_s:
        for (run_id, platform), sub in post.groupby(["run_id", "platform"], sort=True):
            b = _bouts_for_series(
                sub.t_seconds.reset_index(drop=True),
                gap_s=float(gap),
                window=window,
                bin_s=bin_s,
                floor_s=floor_s,
            )
            b.insert(0, "platform", platform)
            b.insert(0, "gap_s", float(gap))
            b.insert(0, "run_id", run_id)
            frames.append(b)

    if not frames:
        return pd.DataFrame(columns=BOUT_COLUMNS)
    return pd.concat(frames, ignore_index=True)[BOUT_COLUMNS]


def in_window_bouts(bouts: pd.DataFrame, gap_s: float = PRIMARY_GAP_S, platform: str | None = None) -> pd.DataFrame:
    """Bouts at ``gap_s`` lying wholly inside the window, optionally on one platform.

    The universe the effort draws, the pose aggregation and the QC covariates
    all share -- they are joined on ``bout_id``, so they must agree.
    """
    m = (bouts.gap_s == gap_s) & bouts.in_window
    if platform is not None:
        m &= bouts.platform == platform
    return bouts[m]


def build_exposure_table(
    events: pd.DataFrame,
    runs: pd.DataFrame,
    *,
    blind_s_per_trigger: float,
    window: tuple[float, float] = (10.0, 2900.0),
    bin_s: float = 600.0,
) -> pd.DataFrame:
    """Per-run, per-bin observed time: wall time in the window minus blind time.

    Resolved per bin rather than per run because the blind-time loss
    concentrates in whichever bins had the most triggers, and ``bin_slope``
    needs each bin's own rate.
    """
    lo, hi = window
    n_bins = int(np.ceil(hi / bin_s))
    dur = runs.set_index("run_id").actual_duration_s.astype(float)

    post = events.loc[~events.is_artifact & events.t_seconds.between(lo, hi)]
    trig = (
        post.assign(time_bin=ev_mod.time_bin(post.t_seconds, bin_s))
        .groupby(["run_id", "time_bin"])
        .size()
        .rename("n_triggers_in_bin")
    )

    rows = []
    for run_id in runs.run_id:
        d = float(dur.get(run_id, 0.0))
        end = min(hi, d)
        for b in range(n_bins):
            b0, b1 = b * bin_s, (b + 1) * bin_s
            wall = max(0.0, min(b1, end) - max(b0, lo))
            rows.append(
                {
                    "run_id": run_id,
                    "time_bin": b,
                    "bin_start_s": max(b0, lo),
                    "bin_end_s": min(b1, hi),
                    "wall_s": wall,
                    "n_triggers_in_bin": int(trig.get((run_id, b), 0)),
                    "covers_bin": d >= min(b1, hi),
                }
            )
    out = pd.DataFrame(rows)
    out["observed_time_s"] = out.wall_s - blind_s_per_trigger * out.n_triggers_in_bin
    # A bin can only go blind for time it actually had; the clip guards against
    # a refit constant that overshoots on an extreme run.
    out["observed_time_s"] = out.observed_time_s.clip(lower=0.0)
    return out


def effort_subsample_index(
    bouts: pd.DataFrame,
    run_ids,
    *,
    k: int = 20,
    b: int = 200,
    seed: int = 20260907,
    gap_s: float = PRIMARY_GAP_S,
    platform: str = "active",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialise ``b`` draws of ``k`` bouts per run, without replacement.

    Image-derived features are computed on frames that exist only because a
    trigger fired, so a 120-bout run and a 15-bout run give wildly different
    estimation precision -- and that precision is a direct function of the
    treatment effect. Equalising the draw size removes that.

    Sampling is **without replacement**. Drawing 20 from a run that has 8
    equalises the nominal count but not the effective one, manufacturing false
    precision in exactly the runs where a compound may have suppressed
    responding -- the failure mode the equalisation exists to prevent. Runs
    below ``k`` are marked ``effort_equalized=False`` and keep their full set.

    ``run_ids`` is every run to report on; one with no qualifying bout gets a
    status row with ``n_bouts_available=0``. Runs are drawn in sorted order,
    so the draws do not depend on the order ``run_ids`` is given in.

    Returns ``(index, run_status)``.
    """
    sub = in_window_bouts(bouts, gap_s, platform)
    rng = np.random.default_rng(seed)
    by_run = {r: g.bout_id.to_numpy() for r, g in sub.groupby("run_id", sort=True)}

    idx_rows: list[pd.DataFrame] = []
    status_rows: list[dict] = []
    for run_id in sorted(set(run_ids)):
        ids = by_run.get(run_id, ())
        n = len(ids)
        ok = n >= k
        status_rows.append(
            {"run_id": run_id, "n_bouts_available": n, "effort_equalized": ok, "k": k}
        )
        if not ok:
            continue
        draws = np.stack([rng.choice(ids, size=k, replace=False) for _ in range(b)])
        idx_rows.append(
            pd.DataFrame(
                {
                    "run_id": run_id,
                    "draw": np.repeat(np.arange(b), k),
                    "bout_id": draws.ravel(),
                }
            )
        )

    index = (
        pd.concat(idx_rows, ignore_index=True)
        if idx_rows
        else pd.DataFrame(columns=["run_id", "draw", "bout_id"])
    )
    index["k"] = k
    index["seed"] = seed
    return index, pd.DataFrame(status_rows)

