"""Run-level features from the event TSVs, and the QC covariates.

**Trigger features (Tier A)** come from the event TSVs: bout rates per observed
minute and doses per visit. They are not affected by image sampling.

**Timing features** describe *when* and *in what order* visits happen: the
change in platform preference across the session, switching from active to
inactive, the within-run decline, and the latencies to the first trigger on
each platform.

**QC covariates (Tier C)** are never features. They are tested against the
clusters so that a cluster tracking detection quality, or when in the session
its pose frames were sampled, is flagged as not a phenotype.

Pose features live in :mod:`zfsa.pose_features`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Canonical-frame band along the tank rim (distance from the nearest wall).
RIM_INNER = 0.0
RIM_OUTER = 0.08


def in_rim_band(
    x_tank: np.ndarray, y_tank: np.ndarray, *, inner: float = RIM_INNER, outer: float = RIM_OUTER
) -> np.ndarray:
    """True where a detection sits in the canonical-frame rim band."""
    x = np.asarray(x_tank, dtype=float)
    y = np.asarray(y_tank, dtype=float)
    edge = np.minimum.reduce([x, y, 1.0 - x, 1.0 - y])
    return (edge >= inner) & (edge <= outer)


def assign_bout(t: np.ndarray, t_start: np.ndarray, t_end: np.ndarray) -> np.ndarray:
    """Index of the bout (sorted by ``t_start``) containing each time, or -1."""
    i = np.searchsorted(t_start, t, side="right") - 1
    ok = (i >= 0) & (t <= np.where(i >= 0, t_end[np.clip(i, 0, None)], -np.inf))
    return np.where(ok, i, -1)


# --------------------------------------------------------------------- Tier A


def _run_tier_a(
    b_act: pd.DataFrame,
    b_ina: pd.DataFrame,
    t_act: np.ndarray,
    obs_min: float,
    floor_s: float,
) -> dict:
    out: dict = {}
    n_act, n_ina = len(b_act), len(b_ina)
    out["n_active_bouts_in_window"] = n_act
    out["n_inactive_bouts_in_window"] = n_ina
    out["active_bout_rate_per_obs_min"] = n_act / obs_min if obs_min > 0 else np.nan
    out["inactive_bout_rate_per_obs_min"] = n_ina / obs_min if obs_min > 0 else np.nan
    # Descriptive validator only: many runs have zero inactive bouts, so the
    # per-run ratio is degenerate for a sizeable share of runs.
    out["logodds_active"] = float(np.log((n_act + 0.5) / (n_ina + 0.5)))

    if n_act:
        out["mean_bout_duration_s"] = float(b_act.duration_s.mean())
        out["mean_triggers_per_bout"] = float(b_act.n_triggers.mean())
    else:
        out["mean_bout_duration_s"] = np.nan
        out["mean_triggers_per_bout"] = np.nan

    # Intervals sitting on the pump floor are the pump setting the rate, not the fish.
    n_int = int(b_act.n_triggers.sum() - n_act) if n_act else 0
    out["frac_iei_at_floor"] = float(b_act.n_at_floor.sum() / n_int) if n_int > 0 else np.nan
    out["first_bout_latency_s"] = float(t_act.min()) if len(t_act) else np.nan
    return out


def _bin_slope(b_act: pd.DataFrame, exposure_run: pd.DataFrame) -> float:
    """OLS slope of log active bout rate on time bin -- one parameter for the decay.

    Bins the run did not reach are dropped rather than scored as zero: a run
    that stopped early has no observation there, and calling that a zero rate
    would read a short run as a steep decay.
    """
    counts = b_act.groupby("time_bin").size()
    e = exposure_run[exposure_run.covers_bin & (exposure_run.observed_time_s > 0)]
    if len(e) < 3:
        return np.nan
    y = np.log((e.time_bin.map(counts).fillna(0).to_numpy() + 0.5) / (e.observed_time_s.to_numpy() / 60.0))
    x = e.time_bin.to_numpy(dtype=float)
    if len(np.unique(x)) < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def tier_a(
    bouts: pd.DataFrame,
    events: pd.DataFrame,
    exposure: pd.DataFrame,
    runs: pd.DataFrame,
    *,
    gap_s: float = 5.0,
    floor_s: float = 0.7,
) -> pd.DataFrame:
    """One row per run of event-timing features over the analysis window."""
    b = bouts[(bouts.gap_s == gap_s) & bouts.in_window]
    b_act = {k: v for k, v in b[b.platform == "active"].groupby("run_id")}
    b_ina = {k: v for k, v in b[b.platform == "inactive"].groupby("run_id")}

    win_act = events[(events.platform == "active") & ~events.is_artifact & events.in_window]
    t_by_run = {k: v.t_seconds.to_numpy() for k, v in win_act.groupby("run_id")}
    exp_by_run = {k: v for k, v in exposure.groupby("run_id")}
    empty = bouts.iloc[:0]

    rows = []
    for run_id in runs.run_id:
        ba = b_act.get(run_id, empty)
        bi = b_ina.get(run_id, empty)
        exp = exp_by_run.get(run_id, exposure.iloc[:0])
        obs_min = float(exp.observed_time_s.sum()) / 60.0 if len(exp) else np.nan
        rec = {"run_id": run_id}
        rec.update(_run_tier_a(ba, bi, t_by_run.get(run_id, np.array([])), obs_min, floor_s))
        rec["observed_time_in_window_s"] = float(exp.observed_time_s.sum()) if len(exp) else np.nan
        rec["bin_slope"] = _bin_slope(ba, exp) if len(ba) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- timing


def timing_features(
    runs: pd.DataFrame,
    events: pd.DataFrame,
    bouts: pd.DataFrame,
    exposure: pd.DataFrame,
    tier_a_df: pd.DataFrame,
    *,
    window: tuple[float, float],
    gap_s: float = 5.0,
    late_min_obs_s: float = 300.0,
) -> pd.DataFrame:
    """Order-and-timing features, one row per run.

    * ``pref_change`` -- late (bins >= 3) minus early (bins <= 1) log-odds of
      active vs inactive bouts; NaN when the run has < ``late_min_obs_s`` of
      observed late time.
    * ``switch_excess`` -- log((observed + 0.5) / (chance + 0.5)) count of active
      bouts followed by an inactive bout. A count ratio, because a proportion
      difference piles half the runs onto an exact 0.
    * ``bin_slope`` -- from Tier A.
    * ``first_inactive_latency_log`` -- log(1 + s from the window start to the
      first inactive trigger), censored at the window end when there is none.
    * ``first_active_latency_log`` -- the same for the active platform.
    """
    start, end_window = window
    b = bouts[(bouts.gap_s == gap_s) & bouts.in_window]
    ina_ev = events[(events.platform == "inactive") & ~events.is_artifact & events.in_window]
    first_ina = ina_ev.groupby("run_id").t_seconds.min()
    late_obs = exposure[exposure.time_bin >= 3].groupby("run_id").observed_time_s.sum()
    ta = tier_a_df.set_index("run_id")
    rows = []
    for r in runs.itertuples():
        rid = r.run_id
        bb = b[b.run_id == rid].sort_values("t_start")
        act, ina = bb[bb.platform == "active"], bb[bb.platform == "inactive"]
        rec = {"run_id": rid}

        aE, iE = int((act.time_bin <= 1).sum()), int((ina.time_bin <= 1).sum())
        aL, iL = int((act.time_bin >= 3).sum()), int((ina.time_bin >= 3).sum())
        rec["pref_change"] = (np.log((aL + .5) / (iL + .5)) - np.log((aE + .5) / (iE + .5))
                              if late_obs.get(rid, 0.0) >= late_min_obs_s else np.nan)

        is_ina = (bb.platform == "inactive").to_numpy()
        nA, nI = int((~is_ina).sum()), int(is_ina.sum())
        succ = np.flatnonzero(~is_ina[:-1])
        if len(succ) and nA + nI > 1:
            obs_ai = float(is_ina[succ + 1].sum())
            exp_ai = len(succ) * nI / (nA + nI - 1)
            rec["switch_excess"] = float(np.log((obs_ai + 0.5) / (exp_ai + 0.5)))
        else:
            rec["switch_excess"] = np.nan

        rec["bin_slope"] = float(ta.bin_slope.get(rid, np.nan))
        end = min(float(r.actual_duration_s), end_window)
        t_i = first_ina.get(rid, np.nan)
        rec["first_inactive_censored"] = bool(np.isnan(t_i))
        rec["first_inactive_latency_log"] = float(np.log1p((end if np.isnan(t_i) else t_i) - start))
        rec["first_active_latency_log"] = float(np.log1p(ta.first_bout_latency_s.get(rid) - start))
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- Tier C


def draw_mean_time_bin(index: pd.DataFrame, bouts: pd.DataFrame, *, gap_s: float = 5.0) -> pd.DataFrame:
    """Mean time bin of each run's effort draws -- a QC covariate, not a feature.

    A run whose triggering collapses after the first bin yields pose samples
    drawn almost entirely from early in the session, so any pose difference is
    partly a difference in *when* it was measured.
    """
    b = bouts[(bouts.gap_s == gap_s) & (bouts.platform == "active") & bouts.in_window]
    m = index.merge(b[["run_id", "bout_id", "time_bin"]], on=["run_id", "bout_id"], how="left")
    return m.groupby("run_id").time_bin.mean().rename("subsample_mean_time_bin").reset_index()


def trigger_fish_present_fraction(det: pd.DataFrame, bouts: pd.DataFrame, *, gap_s: float = 5.0) -> pd.DataFrame:
    """Per run: mean over active bouts of the share of detections that are the trigger fish.

    Low values mean the fish that caused the dose was often not identified
    inside the active ROI, which weakens every trigger-fish feature.
    """
    b = bouts[(bouts.gap_s == gap_s) & (bouts.platform == "active") & bouts.in_window]
    rows = []
    for run_id, bb in b.sort_values(["run_id", "t_start"]).groupby("run_id", sort=True):
        dd = det[det.run_id == run_id]
        if dd.empty:
            continue
        i = assign_bout(dd.t_seconds.to_numpy(), bb.t_start.to_numpy(), bb.t_end.to_numpy())
        dd = dd[i >= 0].assign(_b=i[i >= 0])
        if dd.empty:
            continue
        rows.append({"run_id": run_id,
                     "trigger_fish_present_fraction": float(dd.groupby("_b").is_trigger_fish.mean().mean())})
    return pd.DataFrame(rows, columns=["run_id", "trigger_fish_present_fraction"])


def run_detection_qc(frame_qc: pd.DataFrame) -> pd.DataFrame:
    """Per-run detection quality from the per-frame QC table."""
    return (
        frame_qc.groupby("run_id")
        .agg(
            mean_n_detected=("n_detected", "mean"),
            mean_conf_5th=("conf_5th", "mean"),
            mean_max_iou=("max_pairwise_iou", "mean"),
            frac_frames_under_5=("n_kept", lambda s: (s < 5).mean()),
        )
        .reset_index()
    )


def tier_c(
    runs: pd.DataFrame,
    calibration: pd.DataFrame,
    frame_qc: pd.DataFrame,
    det: pd.DataFrame,
    bouts: pd.DataFrame,
    effort_index: pd.DataFrame,
    effort_status: pd.DataFrame,
    *,
    gap_s: float = 5.0,
) -> pd.DataFrame:
    """QC covariates per run. Tested against the clusters, never used as features."""
    out = runs[["run_id", "blind_fraction", "covers_window", "run_order_within_day",
                "actual_duration_s"]].copy()
    out = out.merge(calibration[["run_id", "rms_reprojection_px", "calibration_ok"]], on="run_id", how="left")
    out = out.merge(run_detection_qc(frame_qc), on="run_id", how="left")
    out = out.merge(effort_status[["run_id", "n_bouts_available", "effort_equalized"]], on="run_id", how="left")
    out = out.merge(draw_mean_time_bin(effort_index, bouts, gap_s=gap_s), on="run_id", how="left")
    out = out.merge(trigger_fish_present_fraction(det, bouts, gap_s=gap_s), on="run_id", how="left")
    return out
