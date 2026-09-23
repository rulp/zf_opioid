"""Zebrafish opioid self-administration: run-level PCA -> GMM pipeline.

A trimmed, standalone subset of the lab's analysis code. It takes one folder
of runs plus ``metadata.csv`` to 14 per-run behavioural features (trigger
rates, event timing, and pose features from a YOLOv11-pose model) and a
Gaussian-mixture analysis of those features. Drive it with
``run_pipeline.py``; see README.md.

Modules, in pipeline order: paths, config, naming, metadata, events,
geometry, infer, bouts, features, pose_features, normalize, gmm, analysis.
"""

__all__ = [
    "paths", "config", "naming", "metadata", "events", "geometry", "infer", "bouts",
    "features", "pose_features", "normalize", "gmm", "analysis",
]
