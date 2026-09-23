"""Every tunable in one place.

:data:`DEFAULTS` is the single source of truth. ``params.yaml`` is optional: it
is deep-merged over the defaults, so it only needs the values you want to
change, and deleting it changes nothing.

Resolve the config once, at the entry point, and pass values down as keyword
arguments. Library code never reads the config at call time -- a function
whose behaviour depends on a global read at an unknown moment is not testable.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

#: The 14 PCA input features, in the order they enter the matrix. The order is
#: part of the analysis: it sets the column order for parallel analysis and the
#: GMM, so changing it changes random draws and can change results.
FEATURES: list[str] = [
    "inactive_bout_rate_per_obs_min",
    "nontrig_pref_index",
    "pref_change",
    "active_bout_rate_per_obs_min",
    "mean_nn_distance",
    "switch_excess",
    "mean_triggers_per_bout",
    "polarization_nontrigger",
    "bin_slope",
    "frac_in_rim",
    "first_inactive_latency_log",
    "trig_radial_offset",
    "first_active_latency_log",
    "trig_heading_outward",
]

DEFAULTS: dict[str, Any] = {
    "events": {
        # Events before this are background-model initialisation artifacts.
        "artifact_cutoff_s": 10.0,
        # Triggers closer than this belong to one bout (one visit).
        "bout_gap_s": 5.0,
        # Width of the within-run time bins.
        "bin_s": 600.0,
    },
    "window": {
        # Analysis window. Runs that stop early are kept; rates use observed time.
        "start_s": 10.0,
        "end_s": 2900.0,
    },
    "bouts": {
        # Pump hardware floor between triggers, and the tolerance for "at the floor".
        "refractory_s": 0.6,
        "floor_tol_s": 0.1,
    },
    "blind_time": {
        # The rig cannot detect while the pump fires, so each trigger costs a
        # fraction of a second of observable time. It is refit on the supplied
        # runs; with too few runs, or a poor fit, this rig constant is used.
        "s_per_trigger": 0.188,
        "min_runs_to_fit": 10,
        "min_r_squared": 0.99,
    },
    "effort": {
        # Pose features are averaged over b draws of k active bouts per run, so
        # every run is measured with the same effort. Runs with < k bouts are
        # not scored on pose (and so drop out of the PCA).
        "k": 20,
        "b": 200,
        "seed": 20260907,
    },
    "features": {
        # Canonical-frame band along the tank wall, for frac_in_rim.
        "rim_inner": 0.0,
        "rim_outer": 0.08,
        # Late-minus-early preference change needs this much observed late time.
        "late_min_obs_s": 300.0,
    },
    "inference": {
        "weights": "models/best.pt",
        "conf": 0.25,
        "iou": 0.65,
        "batch": 16,
        # "cpu" reproduces the reference detections exactly; "cuda:0" is faster.
        "device": "cpu",
        # Frames sampled per run for the background model used in calibration.
        "background_samples": 120,
    },
    "metadata": {
        # Compounds counted as positive controls when drafting metadata.csv.
        "positive_controls": ["MK801", "ketamine", "methadone", "naloxone"],
        # Folder-name spellings -> canonical compound label, used only when
        # drafting metadata.csv. Keys are lower-case with non-alphanumerics removed.
        "compound_aliases": {
            "dmso": "DMSO",
            "mk801": "MK801",
            "ketamine": "ketamine",
            "naloxone": "naloxone",
            "methadone": "methadone",
            "hydrocodone": "hydrocodone",
            "regularcbd": "CBD",
            "cbd": "CBD",
            "h2cbdnat": "H2CBD_NAT",
            "h2cbdnatural": "H2CBD_NAT",
            "hcbdnat": "H2CBD_NAT",
            "h2cbdsyn": "H2CBD_SYN",
            "h4cbdnat": "H4CBD_NAT",
            "h4cbdnatural": "H4CBD_NAT",
            "h4cbdsyn": "H4CBD_SYN",
            "h2ppsyn": "H2PP_SYN",
            "h2ppsny": "H2PP_SYN",
            "h4ppsyn": "H4PP_SYN",
            "h2posyn": "H2PO_SYN",
        },
    },
    "analysis": {
        "features": FEATURES,
        "seed": 20260907,
        # PCs retained by parallel analysis, capped here.
        "pc_cap": 3,
        "n_parallel_analysis": 1000,
        # GMM grid: K = 1..k_max x covariance types, best of n_init starts.
        "k_max": 5,
        "covariance_types": ["spherical", "diag", "tied", "full"],
        "n_init": 20,
        # K=1 wins unless the best K>=2 model beats it by at least this much BIC.
        "tie_dbic": 2.0,
        # A fit competes only if every cluster holds at least this many runs.
        # 1 = no minimum: a single run may be a cluster if BIC says so.
        "min_cluster_n": 1,
        # Component covariance eigenvalue below this = collapsed to the floor.
        "degenerate_eig": 1.0e-4,
        # Single-Gaussian null replicates, bootstrap resamples, label permutations.
        "n_null": 1000,
        "n_boot": 500,
        "n_perm": 2000,
        # Below this MAD/SD ratio the DMSO MAD describes a mass point, not a spread.
        "mad_sd_min": 0.25,
        # Runs per feature below which the report warns that p is large for n.
        "runs_per_feature": 6,
        # Parallel workers for the null and bootstrap (-1 = all cores).
        "n_jobs": -1,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str | Path | None = None) -> dict[str, Any]:
    """:data:`DEFAULTS` deep-merged with a params YAML file, if given and present."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"params file {p} not found")
    import yaml

    loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{p}: top level must be a mapping, got {type(loaded).__name__}")
    unknown = set(loaded) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"{p}: unknown section(s) {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    return _deep_merge(cfg, loaded)


def hash_config(cfg: dict) -> str:
    """Stable digest of a config dict, for provenance."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


def window(cfg: dict) -> tuple[float, float]:
    return (float(cfg["window"]["start_s"]), float(cfg["window"]["end_s"]))
