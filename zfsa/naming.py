"""Parsers for run folder names and trigger JPEG filenames.

These are used in two places only: to **draft** ``metadata.csv`` from the run
folder names (``make_metadata.py``), and to read the frame index out of each
trigger JPEG's filename. The analysis itself takes its metadata from
``metadata.csv``, never from folder names, so a folder named inconsistently
is fixed by editing the CSV, not by renaming anything.

The subtle part is the group token. ``g2_5`` means two different things:

===========================  ==============================  ==============
pattern                      meaning                         example
===========================  ==============================  ==============
``g<N>_5_fish``              group N, run with five fish     ``g3_5_fish_DMSO_control`` -> g3
``g<N>.5`` / ``g<N>_5_...``  group N.5 (a distinct group)    ``g2.5_DMSO_1``            -> g2.5
===========================  ==============================  ==============

A half group (e.g. group 2 topped up with fish from group 3) has a different
drug history and is not interchangeable with the whole group, so the
``_5_fish`` form is tested *before* the fractional form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

# --- group -----------------------------------------------------------------

_BOUNDARY = r"(?:^|[_. ])"
# Trailing boundary: '\b' is useless here because '_' is a word character, so
# 'fish\b' would not match in 'g3_5_fish_DMSO_control'.
_END = r"(?![A-Za-z0-9])"

# "group N, five fish" -- consume it so the fractional rule cannot see it.
_RE_G_NFISH = re.compile(_BOUNDARY + r"[gG](\d)[._]5[._]fish" + _END)
# a genuine fractional group: gN.5 / gN_5 not followed by another digit
_RE_G_FRAC = re.compile(_BOUNDARY + r"[gG](\d)[._]5(?![0-9])")
_RE_G_PLAIN = re.compile(_BOUNDARY + r"[gG](\d)(?![0-9])")


def parse_group(name: str) -> str | None:
    """Return the fish group id (``"2"``, ``"2.5"``, ...) or None if absent."""
    tail = _strip_stamp(name)
    m = _RE_G_NFISH.search(tail)
    if m:
        return m.group(1)
    m = _RE_G_FRAC.search(tail)
    if m:
        return f"{m.group(1)}.5"
    m = _RE_G_PLAIN.search(tail)
    if m:
        return m.group(1)
    return None


# --- dose ------------------------------------------------------------------

_NUM = r"(\d+(?:\.\d+)?)"
_RE_DOSE_UM = re.compile(_NUM + r"\s*[_ ]?\s*uM\b", re.I)
_RE_DOSE_MGL = re.compile(_NUM + r"\s*[_ ]?\s*mg[ _]?per[ _]?L\b", re.I)
_RE_DOSE_UL = re.compile(_NUM + r"\s*[_ ]?\s*uL\b", re.I)


@dataclass(frozen=True)
class Dose:
    value: float | None
    unit: str | None  # "uM" | "mg/L" | "uL"


def parse_dose(name: str) -> tuple[Dose, str]:
    """Extract the dose and return ``(dose, name_with_dose_removed)``.

    ``uL`` is a *volume*, not a concentration -- it appears where the operator
    recorded what was pipetted rather than the resulting molarity. It is kept
    with its unit rather than silently coerced to uM.
    """
    tail = _strip_stamp(name)
    for rx, unit in ((_RE_DOSE_UM, "uM"), (_RE_DOSE_MGL, "mg/L"), (_RE_DOSE_UL, "uL")):
        m = rx.search(tail)
        if m:
            return Dose(float(m.group(1)), unit), (tail[: m.start()] + tail[m.end() :])
    return Dose(None, None), tail


# --- compound --------------------------------------------------------------

_VEHICLE_TOKENS = {"dmso", "control", "vehicle", "untreated"}

# Boilerplate that carries no compound information. Explicit lookaround
# boundaries: '\b' is wrong here because '_' is a word character.
_RE_NOISE = re.compile(
    _BOUNDARY
    + r"(?:self[_ ]admin(?:istration)?|5[_ ]?fish|5ea|fish|run|rerun|control|test|"
    r"[gG]\d(?:[._]5)?)"
    + _END,
    re.I,
)
_RE_TRAILING_ORD = re.compile(r"(?:^|[_ ])(?:run[_ ]?)?\d{1,2}$")


def normalise_key(text: str) -> str:
    """Lower-case with non-alphanumerics removed: the key used by compound aliases."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_compound(name: str, aliases: dict[str, str] | None = None) -> tuple[str | None, bool]:
    """Return ``(compound_label, is_vehicle)``.

    ``aliases`` maps :func:`normalise_key` spellings to one canonical label, so
    that ``H2CBD_NAT``, ``h2CBD_natural`` and ``H_CBD_NAT`` collapse to one
    condition. A run is vehicle if nothing but DMSO/control/vehicle tokens is
    left once dose and boilerplate are removed.
    """
    aliases = aliases or {"dmso": "DMSO"}
    _, tail = parse_dose(name)
    tail = _RE_NOISE.sub(" ", tail)
    tail = re.sub(r"[_\-.]+", " ", tail)
    tail = re.sub(r"\s+", " ", tail).strip()
    tail = _RE_TRAILING_ORD.sub("", tail).strip()

    if not tail:
        return (None, False)

    key = normalise_key(tail)
    if key in aliases:
        label = aliases[key]
        return (label, label == "DMSO")

    tokens = [t for t in tail.split() if t]
    if tokens and all(t.lower() in _VEHICLE_TOKENS for t in tokens):
        return ("DMSO", True)

    # Numeric compound codes (9533, 8919, ...) and series codes pass through.
    return (re.sub(r"\s+", "_", tail), False)


# --- free-text failure flags ----------------------------------------------

#: Substrings an operator may type into a folder name to record a failure.
_FAILURE_PATTERNS = [
    (re.compile(r"lethal", re.I), "lethal"),
    (re.compile(r"disconnected[_ ]?tube", re.I), "disconnected tube"),
    (re.compile(r"bugged[_ ]?pump", re.I), "bugged pump"),
    (re.compile(r"\bdied?\b|\bdeath\b", re.I), "death"),
]


def parse_failure(name: str) -> str | None:
    """Return a failure reason encoded in the folder name, if any."""
    tail = _strip_stamp(name)
    hits = [label for rx, label in _FAILURE_PATTERNS if rx.search(tail)]
    return "; ".join(hits) if hits else None


# --- timestamps ------------------------------------------------------------

_RE_STAMP = re.compile(r"^(\d{8})_(\d{6})")


def parse_stamp(name: str) -> datetime:
    """Parse the ``YYYYMMDD_HHMMSS`` prefix into a datetime (run start)."""
    m = _RE_STAMP.match(name)
    if not m:
        raise ValueError(f"no timestamp prefix in {name!r}")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def parse_date(name: str) -> date:
    return parse_stamp(name).date()


def _strip_stamp(name: str) -> str:
    return _RE_STAMP.sub("", name).lstrip("_")


# --- trigger JPEG filenames ------------------------------------------------

_RE_IMG = re.compile(r"^active_trigger_(\d+)_frame_(\d+)_(\d{8})_(\d{6})\.jpg$", re.I)


@dataclass(frozen=True)
class TriggerImage:
    trigger_idx: int
    frame_idx: int
    wall_time: datetime


def parse_trigger_image(filename: str) -> TriggerImage | None:
    """Parse ``active_trigger_NNNN_frame_NNNNNN_YYYYMMDD_HHMMSS.jpg``."""
    m = _RE_IMG.match(filename)
    if not m:
        return None
    return TriggerImage(
        trigger_idx=int(m.group(1)),
        frame_idx=int(m.group(2)),
        wall_time=datetime.strptime(m.group(3) + m.group(4), "%Y%m%d%H%M%S"),
    )
