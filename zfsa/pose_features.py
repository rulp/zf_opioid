"""Pose features: per trigger frame, then per bout, then per run.

Every JPEG exists *because a trigger fired*, so pose features are conditioned
on the behaviour being measured and are exploratory. Two rules keep them
honest:

* **Permutation-invariant over the five fish.** ``fish_slot`` is a within-frame
  index, never an identity -- frames are seconds apart and no fish is tracked
  across them. The one labelled subject is the **trigger fish** (the fish inside
  the active ROI); the other fish are pooled.
* **Aggregated per bout, then over equal-effort draws.** A loitering fish
  contributes many near-duplicate frames from one visit, so frames are averaged
  within a bout first. Runs are then summarised as the median, over ``b`` draws
  of ``k`` bouts, of the draw mean -- so a 120-bout run and a 25-bout run are
  measured with the same effort.

Distances are in *isotropic* tank-height units: canonical x is stretched by
640/480 so that a unit is the same length in x and y.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import features, geometry

ISO = 640.0 / 480.0  # canonical x -> isotropic units of tank height
PLAT_HALF_W = geometry.CANONICAL_PLATFORM["w"] / 2
PLAT_HALF_H = geometry.CANONICAL_PLATFORM["h"] / 2
PLAT_HALF_ISO = 0.5 * (PLAT_HALF_W * ISO + PLAT_HALF_H)  # mean platform half-side, isotropic

#: The pose features that enter the PCA.
POSE_FEATURES = [
    "trig_radial_offset",
    "trig_heading_outward",
    "nontrig_pref_index",
    "mean_nn_distance",
    "polarization_nontrigger",
    "frac_in_rim",
]


def frame_table(
    det: pd.DataFrame,
    cal: pd.DataFrame,
    *,
    rim_inner: float = features.RIM_INNER,
    rim_outer: float = features.RIM_OUTER,
) -> pd.DataFrame:
    """Per trigger frame, permutation-invariant features. One row per frame.

    * ``trig_radial_offset`` -- trigger fish's distance from the active ROI
      centre, in platform half-sides (loitering at the centre vs the edge).
    * ``trig_heading_outward`` -- cos(heading - bearing from the platform to the
      fish): +1 heading off the platform, -1 heading onto it.
    * ``nontrig_pref_index`` -- (d_inactive - d_active)/(d_inactive + d_active),
      mean over the non-trigger fish.
    * ``mean_nn_distance`` -- mean nearest-neighbour distance, all fish.
    * ``polarization_nontrigger`` -- length of the mean heading unit vector of
      the non-trigger fish (needs >= 2 of them).
    * ``frac_in_rim`` -- share of the non-trigger fish in the rim band.

    Non-trigger features use only frames where the trigger fish was identified.
    """
    d = det.sort_values(["run_id", "frame_idx", "fish_slot"]).reset_index(drop=True)
    c = cal.set_index("run_id")
    for src, dst in [("active_roi_canon_x", "ax"), ("active_roi_canon_y", "ay"),
                     ("inactive_roi_canon_x", "ix"), ("inactive_roi_canon_y", "iy")]:
        d[dst] = d.run_id.map(c[src])
    if d[["ax", "ay", "ix", "iy"]].isna().any().any():
        bad = sorted(d.loc[d[["ax", "ay", "ix", "iy"]].isna().any(axis=1), "run_id"].unique())
        raise ValueError(f"runs without calibrated ROI centres: {bad}")

    X, Y = d.x_tank.to_numpy() * ISO, d.y_tank.to_numpy()
    head = np.arctan2(d.head_y.to_numpy() - d.tail_y.to_numpy(),
                      (d.head_x.to_numpy() - d.tail_x.to_numpy()) * ISO)
    AX, AY = d.ax.to_numpy() * ISO, d.ay.to_numpy()
    IX, IY = d.ix.to_numpy() * ISO, d.iy.to_numpy()
    d_act = np.hypot(X - AX, Y - AY)
    d_ina = np.hypot(X - IX, Y - IY)
    cos_to_act = np.cos(head - np.arctan2(AY - Y, AX - X))
    rim = features.in_rim_band(d.x_tank, d.y_tank, inner=rim_inner, outer=rim_outer)
    trig = d.is_trigger_fish.to_numpy()

    fid = d.groupby(["run_id", "frame_idx"], sort=False).ngroup().to_numpy()
    slot = d.fish_slot.to_numpy()
    nf = fid.max() + 1
    assert not pd.Series(list(zip(fid, slot))).duplicated().any(), "duplicate fish slot in a frame"
    assert slot.max() < 5

    P = np.full((nf, 5, 2), np.nan)
    P[fid, slot, 0], P[fid, slot, 1] = X, Y
    D = np.hypot(P[:, :, None, 0] - P[:, None, :, 0], P[:, :, None, 1] - P[:, None, :, 1])
    D[:, np.arange(5), np.arange(5)] = np.nan
    with warnings.catch_warnings():  # all-NaN slices are expected (empty slots, lone fish)
        warnings.simplefilter("ignore", RuntimeWarning)
        nn_fish = np.nanmin(D, axis=2)
        mean_nn = np.nanmean(nn_fish, axis=1)

    has_trig = pd.Series(trig).groupby(fid).transform("any").to_numpy()
    nt = ~trig & has_trig  # non-trigger fish in frames where the trigger fish is identified

    rows = pd.DataFrame({
        "fid": fid,
        "trig_radial_offset": np.where(trig, d_act / PLAT_HALF_ISO, np.nan),
        "trig_heading_outward": np.where(trig, -cos_to_act, np.nan),
        "nontrig_pref_index": np.where(nt, (d_ina - d_act) / (d_ina + d_act), np.nan),
        "_cos": np.where(nt, np.cos(head), np.nan),
        "_sin": np.where(nt, np.sin(head), np.nan),
        "_nt": nt.astype(float),
        "frac_in_rim": np.where(nt, rim.astype(float), np.nan),
    })
    g = rows.groupby("fid")
    F = g.mean()
    n_nt = g["_nt"].sum()
    F["polarization_nontrigger"] = np.where(n_nt >= 2, np.hypot(F["_cos"], F["_sin"]), np.nan)
    F = F.drop(columns=["_cos", "_sin", "_nt"])
    F["mean_nn_distance"] = mean_nn[F.index]
    F["has_trigger_fish"] = pd.Series(has_trig).groupby(fid).first().loc[F.index].to_numpy()
    meta = d.groupby(fid)[["run_id", "frame_idx", "t_seconds"]].first()
    return meta.join(F).reset_index(drop=True)


def aggregate_runs(
    frames: pd.DataFrame,
    cols: list[str],
    bouts: pd.DataFrame,
    index: pd.DataFrame,
    *,
    gap_s: float = 5.0,
) -> pd.DataFrame:
    """Frame -> bout mean -> draw mean -> median across draws.

    Also returns ``<feature>_mcsd``, the SD across draws: a per-run precision
    estimate. A large value means ``k`` bouts do not pin that run down.
    """
    b = bouts[(bouts.gap_s == gap_s) & (bouts.platform == "active") & bouts.in_window]
    parts = []
    for run_id, bb in b.sort_values(["run_id", "t_start"]).groupby("run_id", sort=True):
        ff = frames[frames.run_id == run_id]
        if ff.empty:
            continue
        i = features.assign_bout(ff.t_seconds.to_numpy(), bb.t_start.to_numpy(), bb.t_end.to_numpy())
        ff = ff[i >= 0].assign(bout_id=bb.bout_id.to_numpy()[i[i >= 0]])
        parts.append(ff.groupby("bout_id")[cols].mean().assign(run_id=run_id).reset_index())
    bout_means = pd.concat(parts, ignore_index=True)
    merged = index[["run_id", "draw", "bout_id"]].merge(bout_means, on=["run_id", "bout_id"], how="left")
    per_draw = merged.groupby(["run_id", "draw"])[cols].mean()
    point = per_draw.groupby("run_id").median()
    spread = per_draw.groupby("run_id").std(ddof=1).add_suffix("_mcsd")
    return point.join(spread).reset_index()
