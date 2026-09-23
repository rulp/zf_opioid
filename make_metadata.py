"""Draft metadata.csv from the run folder names, for you to review.

    python make_metadata.py --runs D:/my_runs
    python make_metadata.py --runs D:/my_runs --out D:/my_runs/metadata.csv

Reads each run folder's name (e.g. ``20260303_165830_self_admin_g1_MK801_20uM``)
and pre-fills fish group, compound, dose and vehicle status. It prints every
name it could not fully read. **Open the CSV and check it before running the
pipeline** -- the analysis trusts this file, not the folder names. The columns
are documented in ``zfsa/metadata.py`` and in README.md.

An existing file is never overwritten unless you pass ``--overwrite``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from zfsa import config, metadata, paths  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, type=Path, help="folder holding one subfolder per run")
    ap.add_argument("--out", type=Path, default=None, help="where to write (default: metadata.csv inside --runs)")
    ap.add_argument("--params", type=Path, default=None,
                    help="params YAML with compound aliases / positive controls (default: params.yaml here)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing file")
    args = ap.parse_args(argv)

    params = args.params or (paths.PROJECT_ROOT / "params.yaml")
    cfg = config.load(params if params.exists() else None)
    out = (args.out or (args.runs / "metadata.csv")).resolve()
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to replace it, or --out elsewhere")

    table, issues = metadata.draft(args.runs, aliases=cfg["metadata"]["compound_aliases"],
                                   positive_controls=cfg["metadata"]["positive_controls"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    n_veh = int((table.is_vehicle == "true").sum())
    print(f"wrote {out}")
    print(f"  {len(table)} runs: {n_veh} vehicle, {table.compound[table.is_vehicle != 'true'].nunique()} compounds, "
          f"{table.date.nunique()} days")
    print("  compounds found: " + ", ".join(sorted(set(table.compound) - {""})))
    if issues:
        print(f"\n{len(issues)} thing(s) to check by hand:")
        for i in issues:
            print(f"  - {i}")
    print("\nReview every row (especially compound spellings, fish_group, exclude and flag), then run:")
    print(f"  python run_pipeline.py --runs {args.runs} --metadata {out} --out out")


if __name__ == "__main__":
    main()
