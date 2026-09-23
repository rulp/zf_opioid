"""Event-table reading, bout collapsing, binning, and the blind-time model.

Three corrections live here, all of which change trigger counts:

**The first ~10 s are artifacts.** The detector's background model initialises
with a high baseline that collapses within seconds, firing both ROIs on nearly
every frame. Those events are not behaviour.

**Raw triggers are not independent.** The pump has a 0.6 s refractory floor, so
a single fish loitering over the platform is recorded as several
self-administrations. Collapsing to bouts at a gap threshold separates
"how often did a fish approach" from "how long did it stay".

**The rig goes blind while pumping.** Across runs,
``total_frames = fps * duration - blind_frames_per_trigger * n_triggers``
fits with R^2 ~ 0.999: each trigger costs ~0.2 s during which no detection can
occur. The loss scales with the endpoint itself, so it compresses
treated-vs-control differences. Rates therefore use *observed* time, not
wall-clock duration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

EVENT_COLUMNS = ["frame_idx", "t_seconds", "signal", "baseline"]


def read_events(run_dir: Path, kind: str) -> pd.DataFrame:
    """Read one event TSV. ``kind`` is ``"active"`` or ``"inactive"``."""
    fname = paths.ACTIVE_TSV if kind == "active" else paths.INACTIVE_TSV
    df = pd.read_csv(run_dir / fname, sep="\t")
    missing = set(EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{run_dir / fname}: missing columns {sorted(missing)}")
    df = df.astype(
        {"frame_idx": "int64", "t_seconds": "float64", "signal": "float64", "baseline": "float64"}
    )
    df["platform"] = kind
    return df


def read_run_events(run_dir: Path) -> pd.DataFrame:
    """Both platforms' events for one run, sorted by time."""
    parts = [read_events(run_dir, k) for k in ("active", "inactive")]
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["t_seconds", "platform"], kind="stable").reset_index(drop=True)


def assign_bouts(t_seconds: pd.Series, gap_s: float) -> pd.Series:
    """Label consecutive triggers separated by < ``gap_s`` as one bout.

    Returns 0-based bout ids within the given series. The caller passes a
    single run and a single platform -- bouts are not meaningful across
    platforms.
    """
    if len(t_seconds) == 0:
        return pd.Series([], dtype="int64", index=t_seconds.index)
    t = t_seconds.to_numpy()
    new_bout = np.empty(len(t), dtype=bool)
    new_bout[0] = True
    new_bout[1:] = np.diff(t) >= gap_s
    return pd.Series(np.cumsum(new_bout) - 1, index=t_seconds.index, dtype="int64")


def time_bin(t_seconds: pd.Series, bin_s: float) -> pd.Series:
    """0-based index of the fixed-width time bin each event falls in."""
    return (t_seconds // bin_s).astype("int64")


@dataclass(frozen=True)
class BlindTimeModel:
    """Least-squares fit of ``last_frame = fps * duration - k * n_triggers``."""

    fps: float
    blind_frames_per_trigger: float
    r_squared: float
    n_runs: int

    @property
    def blind_s_per_trigger(self) -> float:
        return self.blind_frames_per_trigger / self.fps

    def __str__(self) -> str:
        return (
            f"total_frames = {self.fps:.3f} * duration_s "
            f"- {self.blind_frames_per_trigger:.3f} * n_triggers   "
            f"(R^2={self.r_squared:.4f}, n={self.n_runs})"
        )


def fit_blind_time(
    duration_s: pd.Series, last_frame_idx: pd.Series, n_triggers: pd.Series
) -> BlindTimeModel:
    """Fit the frame-loss model across runs.

    No intercept, on purpose: a zero-length run yields zero frames, and a free
    intercept would absorb the very effect being measured.
    """
    mask = duration_s.notna() & (duration_s > 0)
    d = duration_s[mask].to_numpy(dtype=float)
    y = last_frame_idx[mask].to_numpy(dtype=float)
    n = n_triggers[mask].to_numpy(dtype=float)

    X = np.column_stack([d, -n])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return BlindTimeModel(
        fps=float(coef[0]),
        blind_frames_per_trigger=float(coef[1]),
        r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        n_runs=int(mask.sum()),
    )


def summarise_run(
    run_dir: Path,
    *,
    window: tuple[float, float],
    bout_gap_s: float,
    artifact_cutoff_s: float,
) -> dict:
    """Per-run event counts, raw and corrected, plus duration and last frame."""
    ev = read_run_events(run_dir)
    act = ev[ev.platform == "active"]
    ina = ev[ev.platform == "inactive"]

    out: dict = {
        "n_active_raw": len(act),
        "n_inactive_raw": len(ina),
        "actual_duration_s": float(ev.t_seconds.max()) if len(ev) else 0.0,
        "last_frame_idx": int(ev.frame_idx.max()) if len(ev) else 0,
        "n_triggers_total": len(ev),
    }

    post = ev[ev.t_seconds >= artifact_cutoff_s]
    out["n_active_post_artifact"] = int((post.platform == "active").sum())
    out["n_inactive_post_artifact"] = int((post.platform == "inactive").sum())

    lo, hi = window
    win = ev[(ev.t_seconds >= lo) & (ev.t_seconds <= hi)]
    for kind in ("active", "inactive"):
        sub = win[win.platform == kind]
        out[f"n_{kind}_in_window"] = len(sub)
        out[f"n_{kind}_bouts_in_window"] = (
            int(assign_bouts(sub.t_seconds, bout_gap_s).nunique()) if len(sub) else 0
        )
    out["covers_window"] = out["actual_duration_s"] >= hi
    return out


def build_events_table(
    run_dirs: dict[str, Path],
    *,
    window: tuple[float, float],
    bout_gap_s: float,
    bin_s: float,
    artifact_cutoff_s: float,
) -> pd.DataFrame:
    """One row per event across all runs, with bout, bin and window flags."""
    frames = []
    for run_id, d in run_dirs.items():
        ev = read_run_events(d)
        if ev.empty:
            continue
        ev.insert(0, "run_id", run_id)
        ev["is_artifact"] = ev.t_seconds < artifact_cutoff_s
        ev["in_window"] = ev.t_seconds.between(*window)
        ev["time_bin"] = time_bin(ev.t_seconds, bin_s)
        # Bouts are computed on post-artifact events only; artifact-window
        # events would otherwise glue the start of the run into one huge bout.
        ev["bout_id"] = pd.NA
        for kind in ("active", "inactive"):
            m = (ev.platform == kind) & ~ev.is_artifact
            if m.any():
                ev.loc[m, "bout_id"] = assign_bouts(ev.loc[m, "t_seconds"], bout_gap_s)
        ev["bout_id"] = ev["bout_id"].astype("Int64")
        frames.append(ev)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
