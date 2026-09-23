"""Where the data are and where outputs go.

Input is **one folder of runs**. Each run is a subfolder whose name starts with
the run's start time, ``YYYYMMDD_HHMMSS``, and that holds::

    active_events.tsv      one row per active-platform trigger
    inactive_events.tsv    one row per inactive-platform trigger
    params.json            rig settings, including both hand-drawn ROIs
    images/                one JPEG per active trigger

The 15-character timestamp prefix is the run's id everywhere downstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ACTIVE_TSV = "active_events.tsv"
INACTIVE_TSV = "inactive_events.tsv"
PARAMS_JSON = "params.json"
IMAGES_SUBDIR = "images"
REQUIRED_FILES = (ACTIVE_TSV, INACTIVE_TSV, PARAMS_JSON)

#: Timestamp prefix length, ``YYYYMMDD_HHMMSS``.
STAMP_LEN = 15

#: The zf_opioid directory (parent of this package), for resolving relative paths
#: such as the model weights in params.yaml.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Project:
    """Input and output locations for one pipeline run."""

    runs_dir: Path
    out_dir: Path

    @property
    def cache_dir(self) -> Path:
        """Per-run intermediates (backgrounds, calibration, pose) reused across runs."""
        return self.out_dir / "cache"

    @property
    def analysis_dir(self) -> Path:
        return self.out_dir / "pca_gmm"

    def ensure_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def resolve(path: str | os.PathLike[str]) -> Path:
    """An absolute path; relative paths are taken from the zf_opioid directory."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def run_stamp(path: str | os.PathLike[str]) -> str:
    """The ``YYYYMMDD_HHMMSS`` run id at the start of a run folder name."""
    return Path(path).name[:STAMP_LEN]


def discover_runs(runs_dir: Path) -> dict[str, Path]:
    """Every run folder under ``runs_dir``, keyed by run id.

    Folders whose names do not start with a ``YYYYMMDD_HHMMSS`` timestamp are
    ignored. Two folders with the same timestamp are an error, because the
    timestamp is the run's identity.
    """
    from . import naming

    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs folder {runs_dir} not found")
    out: dict[str, Path] = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            naming.parse_stamp(d.name)
        except ValueError:
            continue
        stamp = run_stamp(d)
        if stamp in out:
            raise ValueError(f"two run folders share timestamp {stamp}: {out[stamp].name} and {d.name}")
        out[stamp] = d
    if not out:
        raise FileNotFoundError(f"no run folders (named YYYYMMDD_HHMMSS_...) found in {runs_dir}")
    return out


def missing_files(run_dir: Path) -> list[str]:
    """Required files or folders absent from a run folder."""
    miss = [f for f in REQUIRED_FILES if not (run_dir / f).is_file()]
    if not (run_dir / IMAGES_SUBDIR).is_dir():
        miss.append(IMAGES_SUBDIR + "/")
    return miss


def images_dir(run_dir: Path) -> Path:
    return run_dir / IMAGES_SUBDIR


def trigger_images(run_dir: Path) -> list[Path]:
    """Active-trigger JPEGs for a run, in filename (== chronological) order."""
    d = images_dir(run_dir)
    return sorted(d.glob("*.jpg")) if d.is_dir() else []
