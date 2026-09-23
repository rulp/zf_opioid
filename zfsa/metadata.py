"""``metadata.csv``: what was run, in each run folder.

The analysis takes every fact about a run that is not in its TSVs from this
file: fish group, compound, dose, whether it is a vehicle run, whether to drop
or flag it. :func:`draft` pre-fills the file from the run folder names so that
most rows need no editing; review it before running the pipeline.

Columns
-------
``run_id``                 ``YYYYMMDD_HHMMSS``, the start of the run folder name. Required.
``folder``                 run folder name (information only).
``date``                   experiment day, ``YYYY-MM-DD``. Blank = taken from ``run_id``.
``fish_group``             fish group id (``1``, ``2``, ``2.5``, ...). Blank = filled from
                           same-day runs if they agree, else ``unknown_<date>``.
``compound``               compound label; runs sharing a label and dose are replicates.
``dose_value``             number, or blank.
``dose_unit``              ``uM``, ``mg/L``, ``uL`` ..., or blank.
``is_vehicle``             true for vehicle (DMSO) control runs. Required. At least 3 needed.
``is_positive_control``    true for positive-control compounds (marked on figures).
``exclude``                true drops the run before anything is computed (e.g. a fish
                           died, tube disconnected).
``exclusion_reason``       why, for the record.
``flag``                   free text; a non-blank flag keeps the run but marks it on
                           figures and drives a "drop flagged runs" sensitivity refit
                           (e.g. ``bugged pump``).
``run_order_within_day``   1, 2, 3 ...; blank = ranked from the run start times.
``notes``                  anything else (ignored).

Booleans accept true/false, yes/no, 1/0 (blank = false).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import naming, paths

COLUMNS = [
    "run_id", "folder", "date", "fish_group", "compound", "dose_value", "dose_unit",
    "is_vehicle", "is_positive_control", "exclude", "exclusion_reason", "flag",
    "run_order_within_day", "notes",
]
REQUIRED = ["run_id", "compound", "is_vehicle"]
BOOL_COLUMNS = ["is_vehicle", "is_positive_control", "exclude"]
MIN_VEHICLE_RUNS = 3

_TRUE = {"true", "t", "yes", "y", "1", "1.0"}
_FALSE = {"false", "f", "no", "n", "0", "0.0", ""}


class MetadataError(ValueError):
    """metadata.csv does not describe the run folders correctly."""


# --------------------------------------------------------------------- draft


def draft(runs_dir: Path, *, aliases: dict[str, str], positive_controls: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Pre-fill metadata from run folder names. Returns ``(table, issues)``.

    ``issues`` lists what could not be read from a name and needs a human:
    no group, no compound, a dose recorded as a volume, or a failure word.
    """
    runs = paths.discover_runs(runs_dir)
    pos = {c.lower() for c in positive_controls}
    rows, issues = [], []
    for run_id, d in runs.items():
        name = d.name
        group = naming.parse_group(name)
        compound, is_vehicle = naming.parse_compound(name, aliases)
        dose, _ = naming.parse_dose(name)
        failure = naming.parse_failure(name)
        rows.append({
            "run_id": run_id,
            "folder": name,
            "date": naming.parse_date(name).isoformat(),
            "fish_group": group or "",
            "compound": compound or "",
            "dose_value": "" if dose.value is None or is_vehicle else f"{dose.value:g}",
            "dose_unit": "" if dose.unit is None or is_vehicle else dose.unit,
            "is_vehicle": "true" if is_vehicle else "false",
            "is_positive_control": "true" if (compound or "").lower() in pos else "false",
            "exclude": "false",
            "exclusion_reason": "",
            "flag": failure or "",
            "run_order_within_day": "",
            "notes": "",
        })
        if group is None:
            issues.append(f"{name}: no fish group in the name (left blank; resolved from same-day runs)")
        if compound is None:
            issues.append(f"{name}: no compound found in the name -- fill in 'compound'")
        if dose.unit == "uL":
            issues.append(f"{name}: dose recorded as a volume ({dose.value:g} uL), not a concentration")
        if failure:
            issues.append(f"{name}: name mentions '{failure}' -- put in 'flag' (kept); "
                          "set exclude=true if the run must be dropped")
    return pd.DataFrame(rows, columns=COLUMNS), issues


# ---------------------------------------------------------------------- load


def _parse_bool(col: str, s: pd.Series) -> pd.Series:
    v = s.fillna("").astype(str).str.strip().str.lower()
    bad = ~v.isin(_TRUE | _FALSE)
    if bad.any():
        raise MetadataError(f"column '{col}' has values that are not true/false: "
                            f"{sorted(set(s[bad].astype(str)))[:5]}")
    return v.isin(_TRUE)


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def load(csv_path: Path, runs_dir: Path) -> tuple[pd.DataFrame, dict[str, Path], list[str]]:
    """Read and validate ``metadata.csv`` against the run folders.

    Returns ``(runs, run_dirs, warnings)``: ``runs`` is one row per run that is
    *not* excluded, sorted by ``run_id``; ``run_dirs`` maps those run ids to
    their folders. Raises :class:`MetadataError` on anything that would make
    the analysis wrong rather than merely incomplete.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise MetadataError(f"{csv_path} not found. Draft one with: python make_metadata.py --runs {runs_dir}")
    m = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    m.columns = [c.strip() for c in m.columns]
    missing_cols = [c for c in REQUIRED if c not in m.columns]
    if missing_cols:
        raise MetadataError(f"{csv_path.name} is missing required column(s): {missing_cols}")
    for c in COLUMNS:
        if c not in m.columns:
            m[c] = ""
    warnings: list[str] = []

    m["run_id"] = m.run_id.str.strip()
    dup = m.run_id[m.run_id.duplicated()]
    if len(dup):
        raise MetadataError(f"run_id listed more than once: {sorted(set(dup))}")
    bad_id = m.run_id[~m.run_id.str.fullmatch(r"\d{8}_\d{6}")]
    if len(bad_id):
        raise MetadataError(f"run_id must look like YYYYMMDD_HHMMSS; got {list(bad_id)[:5]}")

    folders = paths.discover_runs(runs_dir)
    no_row = sorted(set(folders) - set(m.run_id))
    no_folder = sorted(set(m.run_id) - set(folders))
    if no_row:
        raise MetadataError(f"{len(no_row)} run folder(s) have no row in {csv_path.name}: {no_row[:10]}"
                            " -- add them, or set exclude=true")
    if no_folder:
        raise MetadataError(f"{len(no_folder)} row(s) in {csv_path.name} match no run folder: {no_folder[:10]}")

    for c in BOOL_COLUMNS:
        m[c] = _parse_bool(c, m[c])
    m["run_start"] = m.run_id.map(lambda r: naming.parse_stamp(r))
    stamp_date = m.run_start.dt.strftime("%Y-%m-%d")
    given = pd.to_datetime(m.date.where(~_blank(m.date)), errors="coerce")
    if (given.isna() & ~_blank(m.date)).any():
        raise MetadataError(f"unreadable date(s): {list(m.date[given.isna() & ~_blank(m.date)])[:5]}")
    m["date"] = given.dt.strftime("%Y-%m-%d").where(given.notna(), stamp_date)
    off = m.date != stamp_date
    if off.any():
        warnings.append(f"{int(off.sum())} run(s) have a date different from their folder timestamp: "
                        f"{list(m.run_id[off])[:5]}")

    m["compound"] = m.compound.str.strip()
    m.loc[m.is_vehicle & _blank(m.compound), "compound"] = "DMSO"
    nocomp = ~m.is_vehicle & _blank(m.compound) & ~m.exclude
    if nocomp.any():
        raise MetadataError(f"non-vehicle run(s) with no compound: {list(m.run_id[nocomp])[:10]}")
    dv = pd.to_numeric(m.dose_value.where(~_blank(m.dose_value)), errors="coerce")
    if (dv.isna() & ~_blank(m.dose_value)).any():
        raise MetadataError(f"dose_value is not a number: {list(m.dose_value[dv.isna() & ~_blank(m.dose_value)])[:5]}")
    m["dose_value"] = dv
    m["dose_unit"] = m.dose_unit.str.strip().replace("", np.nan)
    m["fish_group_id"] = m.fish_group.str.strip().replace("", np.nan)
    m["flag"] = m.flag.str.strip()
    m["exclusion_reason"] = m.exclusion_reason.str.strip()

    # Order within the day, counting every listed run (excluded ones included):
    # it describes what the fish and the rig had already been through.
    derived = m.groupby("date").run_start.rank(method="first").astype(int)
    ro = pd.to_numeric(m.run_order_within_day.where(~_blank(m.run_order_within_day)), errors="coerce")
    if (ro.isna() & ~_blank(m.run_order_within_day)).any():
        raise MetadataError("run_order_within_day must be a whole number or blank")
    m["run_order_within_day"] = ro.fillna(derived).astype(int)

    kept = m[~m.exclude].sort_values("run_id").reset_index(drop=True)
    if m.exclude.any():
        warnings.append(f"{int(m.exclude.sum())} run(s) excluded by metadata: "
                        + "; ".join(f"{r.run_id} ({r.exclusion_reason or 'no reason given'})"
                                    for r in m[m.exclude].itertuples()))
    n_veh = int(kept.is_vehicle.sum())
    if n_veh < MIN_VEHICLE_RUNS:
        raise MetadataError(f"only {n_veh} vehicle run(s) after exclusions; scaling and day-centring "
                            f"use the vehicle runs as reference and need at least {MIN_VEHICLE_RUNS}")
    if kept.fish_group_id.isna().any():
        warnings.append(f"{int(kept.fish_group_id.isna().sum())} run(s) have no fish_group; "
                        "they are resolved from same-day runs where those agree")

    cols = ["run_id", "run_start", "date", "fish_group_id", "compound", "dose_value", "dose_unit",
            "is_vehicle", "is_positive_control", "flag", "run_order_within_day"]
    run_dirs = {r: folders[r] for r in kept.run_id}
    return kept[cols], run_dirs, warnings


def build_conditions(runs: pd.DataFrame, vehicle_label: str = "DMSO") -> pd.DataFrame:
    """Add ``condition`` = compound@dose (vehicle runs all get ``vehicle_label``)."""
    out = runs.copy()
    dose = out.dose_value.map(lambda v: "" if pd.isna(v) else f"{v:g}")
    unit = out.dose_unit.fillna("")
    out["condition"] = np.where(out.is_vehicle, vehicle_label, out.compound.astype(str) + "@" + dose + unit)
    out["n_runs_condition"] = out.groupby("condition").run_id.transform("size")
    return out
