"""Group resolution, feature transforms, day-centring and scaling.

**Day-centring is empirical-Bayes shrinkage, not subtraction.** Most days have
one or two vehicle runs, and subtracting a single same-day control forms a
difference whose variance is roughly twice the residual -- more noise than the
day term it removes. Instead the day offset estimated from that day's controls
is scaled by ``n_j / (n_j + sigma_e^2 / sigma_d^2)``, which for a one-control
day correctly does most of nothing.

**Scale from the controls only.** A scale estimated from treated runs would
let a compound with a large effect inflate its own denominator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: MAD -> SD for a normal distribution.
MAD_TO_SD = 0.6745


def resolve_group(runs: pd.DataFrame) -> pd.DataFrame:
    """Fill missing fish-group labels without guessing.

    Group is normally constant within a day, so a run with no group on a day
    whose labelled runs all agree is *determined*, not imputed. A day with no
    labels (or conflicting ones) gets its own ``unknown_<date>`` level rather
    than being folded into some group, which would assert a fish history that
    was never recorded.
    """
    out = runs.copy()
    out["fish_group_resolved"] = out.fish_group_id.astype("object")
    out["group_resolution"] = np.where(out.fish_group_id.notna(), "labeled", None)

    for date, g in out.groupby("date"):
        known = sorted(set(g.fish_group_id.dropna()))
        missing = g.fish_group_id.isna()
        if not missing.any():
            continue
        if len(known) == 1:
            out.loc[g.index[missing], "fish_group_resolved"] = known[0]
            out.loc[g.index[missing], "group_resolution"] = "same_day"
        else:
            label = f"unknown_{pd.Timestamp(date).strftime('%Y%m%d')}"
            out.loc[g.index[missing], "fish_group_resolved"] = label
            out.loc[g.index[missing], "group_resolution"] = "unknown_day"
    out["fish_group_resolved"] = out.fish_group_resolved.astype(str)
    return out


def transform(df: pd.DataFrame, spec: dict[str, str]) -> pd.DataFrame:
    """Apply a fixed per-feature transform (``log1p``, ``logit`` or ``identity``).

    Never fit a transform to the data: a fitted transform fits the noise at
    this sample size and makes a feature's meaning depend on the sample.
    """
    out = df.copy()
    for col, how in spec.items():
        if col not in out.columns:
            continue
        v = out[col].astype(float)
        if how == "log1p":
            out[col] = np.log1p(v.clip(lower=0))
        elif how == "logit":
            eps = 1e-3
            p = v.clip(eps, 1 - eps)
            out[col] = np.log(p / (1 - p))
        elif how != "identity":
            raise ValueError(f"unknown transform {how!r} for {col!r}")
    return out


def _variance_components(y: pd.Series, date: pd.Series, is_vehicle: pd.Series):
    """``(sigma2_e, sigma2_d)`` from the control runs alone."""
    d = pd.DataFrame({"y": y, "date": date})[is_vehicle.to_numpy()].dropna()
    if d.empty:
        return np.nan, np.nan
    by_day = d.groupby("date").y
    counts = by_day.size()

    multi = counts[counts >= 2].index
    if len(multi):
        # Pooled within-day variance from the days with replication.
        ss, df_ = 0.0, 0
        for day in multi:
            v = d.loc[d.date == day, "y"].to_numpy()
            ss += ((v - v.mean()) ** 2).sum()
            df_ += len(v) - 1
        sigma2_e = ss / df_ if df_ > 0 else np.nan
    else:
        sigma2_e = np.nan

    means = by_day.mean()
    n_bar = counts.mean()
    sigma2_d = means.var(ddof=1) - (sigma2_e / n_bar if np.isfinite(sigma2_e) else 0.0)
    sigma2_d = max(float(sigma2_d), 0.0)
    return sigma2_e, sigma2_d


def eb_day_center(y: pd.Series, date: pd.Series, is_vehicle: pd.Series) -> tuple[pd.Series, dict]:
    """Subtract a shrunken day offset estimated from that day's controls.

    Needs at least one day with two or more vehicle runs to estimate the
    residual variance; without one, no centring is applied (weight 0).
    """
    sigma2_e, sigma2_d = _variance_components(y, date, is_vehicle)
    d = pd.DataFrame({"y": y, "date": date})
    ctrl = d[is_vehicle.to_numpy()].dropna()
    if ctrl.empty or not np.isfinite(sigma2_e):
        return y, {"weight": 0.0, "sigma2_e": sigma2_e, "sigma2_d": sigma2_d, "n_days": 0}

    grand = ctrl.y.mean()
    by_day = ctrl.groupby("date").y.agg(["mean", "size"])
    if sigma2_d <= 0:
        weights = pd.Series(0.0, index=by_day.index)
    else:
        weights = by_day["size"] / (by_day["size"] + sigma2_e / sigma2_d)
    delta = (by_day["mean"] - grand) * weights

    centered = y - date.map(delta).fillna(0.0)
    info = {
        "weight": float(weights[by_day["size"] == 1].mean()) if (by_day["size"] == 1).any() else float(weights.mean()),
        "sigma2_e": float(sigma2_e),
        "sigma2_d": float(sigma2_d),
        "n_days": int(len(by_day)),
    }
    return centered, info


def scale_guarded(
    F: pd.DataFrame, runs: pd.DataFrame, spec: dict[str, str], *, mad_sd_min: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Transform -> EB day-centre -> scale by the vehicle runs' robust spread.

    Returns ``(Z, scale_report, centring_report)``. The scale is the vehicle MAD,
    except where the MAD is below ``mad_sd_min`` x the vehicle SD: then the MAD
    describes a mass point (e.g. many runs with no inactive bouts) rather than
    the spread, dividing by it sends ordinary runs to |z| of 50-90, and the
    vehicle SD is used instead.
    """
    df = runs[["run_id", "date", "is_vehicle"]].merge(F, on="run_id", how="left")
    feats = [c for c in F.columns if c != "run_id"]
    X = transform(df[feats], spec)
    cent = []
    for c in feats:
        X[c], info = eb_day_center(X[c], df.date, df.is_vehicle)
        cent.append({"feature": c, **info})
    veh = df.is_vehicle.to_numpy()
    Z, rows = X.copy(), []
    for c in feats:
        ref = X.loc[veh, c].dropna()
        med = float(ref.median())
        mad = float((ref - med).abs().median()) / MAD_TO_SD
        sd = float(ref.std(ddof=1))
        if mad > 1e-9 and not (sd > 0 and mad < mad_sd_min * sd):
            s, how = mad, "mad"
        elif sd > 1e-9:
            s, how = sd, ("dmso_sd (MAD = 0)" if mad <= 1e-9 else f"dmso_sd (MAD < {mad_sd_min:g} x SD)")
        else:
            s, how = 1.0, "fallback_1.0"
        Z[c] = (X[c] - med) / s
        rows.append({"feature": c, "dmso_mad": mad, "dmso_sd": sd, "scale": s, "method": how})
    Z.insert(0, "run_id", df.run_id.to_numpy())
    return Z, pd.DataFrame(rows), pd.DataFrame(cent)
