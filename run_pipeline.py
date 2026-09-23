"""Run the whole pipeline: run folders + metadata.csv -> PCA -> GMM report.

    python run_pipeline.py --runs D:/my_runs --metadata metadata.csv --out out

Stages, in order (each prints what it did and any warnings):

  1. check        match metadata.csv to the run folders; every run has its files
  2. events       event tables, per-run summaries, blind-time fit     -> runs, events
  3. calibration  per-run homography to the canonical tank frame      -> calibration
  4. pose         YOLO pose model over every trigger JPEG             -> pose_detections, pose_frame_qc
  5. features     bouts, exposure, effort draws, trigger / timing / QC features
  6. analysis     PCA -> GMM, null, bootstrap, robustness, report     -> out/pca_gmm/

Calibration and pose results are cached per run in ``<out>/cache/``, so adding
runs to the folder and re-running only processes the new ones. Pose inference
is by far the slowest step (roughly 1 minute per 1,000 JPEGs on a laptop CPU).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from zfsa import (analysis, bouts, config, events, features, geometry, infer,  # noqa: E402
                  metadata, naming, normalize, paths)

STAGES = ["check", "events", "calibration", "pose", "features", "analysis"]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------- stages
def stage_check(project: paths.Project, metadata_csv: Path) -> tuple[pd.DataFrame, dict, list[str]]:
    runs, run_dirs, warns = metadata.load(metadata_csv, project.runs_dir)
    bad = {rid: paths.missing_files(d) for rid, d in run_dirs.items()}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        lines = "\n".join(f"  {run_dirs[k].name}: missing {', '.join(v)}" for k, v in bad.items())
        raise SystemExit(f"run folder(s) missing required files:\n{lines}\n"
                         "Fix the folders, or set exclude=true for them in metadata.csv.")
    # Every active trigger should have its JPEG (joined on frame index). A few
    # missing images only lose pose frames; many mean the wrong folder.
    for rid, d in run_dirs.items():
        ev = events.read_events(d, "active")
        imgs = paths.trigger_images(d)
        parsed = [naming.parse_trigger_image(p.name) for p in imgs]
        unparsed = sum(p is None for p in parsed)
        frames = {p.frame_idx for p in parsed if p is not None}
        no_img = int((~ev.frame_idx.isin(frames)).sum())
        no_ev = len(frames - set(ev.frame_idx))
        if unparsed or no_img > 0.05 * max(len(ev), 1) or no_ev:
            warns.append(f"{d.name}: {len(ev)} active events, {len(imgs)} JPEGs "
                         f"({no_img} events without a JPEG, {no_ev} JPEGs without an event, "
                         f"{unparsed} unreadable JPEG names)")
    return runs, run_dirs, warns


def stage_events(project, cfg, runs, run_dirs, warns) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ev_cfg = cfg["events"]
    window = config.window(cfg)
    summ = pd.DataFrame([
        {"run_id": rid, **events.summarise_run(d, window=window, bout_gap_s=ev_cfg["bout_gap_s"],
                                                artifact_cutoff_s=ev_cfg["artifact_cutoff_s"])}
        for rid, d in run_dirs.items()])
    R = runs.merge(summ, on="run_id", how="left")

    bt = cfg["blind_time"]
    fit = events.fit_blind_time(R.actual_duration_s, R.last_frame_idx, R.n_triggers_total)
    if fit.n_runs >= bt["min_runs_to_fit"] and fit.r_squared >= bt["min_r_squared"]:
        blind_s, source = fit.blind_s_per_trigger, "fitted"
    else:
        blind_s, source = float(bt["s_per_trigger"]), "params.yaml (fit unreliable)"
        warns.append(f"blind-time fit unreliable ({fit}); using blind_time.s_per_trigger = {blind_s}")
    log(f"  blind time: {blind_s * 1000:.0f} ms per trigger ({source}); fit: {fit}")
    R["observed_time_s"] = R.actual_duration_s - blind_s * R.n_triggers_total
    R["blind_fraction"] = 1.0 - R.observed_time_s / R.actual_duration_s.replace(0, np.nan)
    R = normalize.resolve_group(R)
    short = R[~R.covers_window]
    if len(short):
        log(f"  {len(short)} run(s) end before the window end ({window[1]:.0f} s); kept, rates use observed time")

    E = events.build_events_table(run_dirs, window=window, bout_gap_s=ev_cfg["bout_gap_s"],
                                  bin_s=ev_cfg["bin_s"], artifact_cutoff_s=ev_cfg["artifact_cutoff_s"])
    blind = {"s_per_trigger": blind_s, "source": source, "fit": str(fit),
             "fit_s_per_trigger": fit.blind_s_per_trigger, "fit_r_squared": fit.r_squared}
    return R, E, blind


def stage_calibration(project, cfg, run_dirs, force: bool, warns) -> pd.DataFrame:
    n_samples = int(cfg["inference"]["background_samples"])
    rows, done = [], 0
    for rid, d in run_dirs.items():
        cp = project.cache_dir / f"calibration_{rid}.json"
        if cp.exists() and not force:
            rows.append(json.loads(cp.read_text(encoding="utf-8")))
            continue
        if force:
            bgp = project.cache_dir / f"bg_{rid}.npz"
            if bgp.exists():
                bgp.unlink()
        row = geometry.calibrate_run(d, n_samples=n_samples, cache_dir=project.cache_dir)
        cp.write_text(json.dumps(row, default=float), encoding="utf-8")
        rows.append(row)
        done += 1
    log(f"  {done} run(s) calibrated, {len(rows) - done} from cache")
    cal = pd.DataFrame(rows)
    failed = cal[~cal.calibration_ok]
    if len(failed):
        raise SystemExit(f"calibration failed for {list(failed.run_id)}; check the images, "
                         "or set exclude=true for these runs in metadata.csv")
    for r in cal[cal.platform_source == "roi_fallback"].itertuples():
        warns.append(f"{r.run_id}: yellow platform not found in the image; used the hand-drawn active ROI")
    for r in cal[~cal.active_is_yellow].itertuples():
        warns.append(f"{r.run_id}: active ROI does not look yellow -- are active and inactive ROIs swapped?")
    return cal


def stage_pose(project, cfg, run_dirs, cal, force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    inf = cfg["inference"]
    weights = paths.resolve(inf["weights"])
    if not weights.exists():
        raise SystemExit(f"model weights {weights} not found")
    key = {"weights_sha256": sha256(weights), "conf": inf["conf"], "iou": inf["iou"], "device": inf["device"]}
    cal_i = cal.set_index("run_id")
    model = None
    det_parts, qc_parts, todo = [], [], []
    for rid, d in run_dirs.items():
        dp, qp, kp = (project.cache_dir / f"pose_{rid}.parquet", project.cache_dir / f"poseqc_{rid}.parquet",
                      project.cache_dir / f"pose_{rid}.json")
        n_img = len(paths.trigger_images(d))
        fresh = (not force and dp.exists() and qp.exists() and kp.exists()
                 and json.loads(kp.read_text()) == {**key, "n_images": n_img})
        if fresh:
            det_parts.append(pd.read_parquet(dp))
            qc_parts.append(pd.read_parquet(qp))
        else:
            todo.append((rid, d, dp, qp, kp, n_img))
    log(f"  {len(todo)} run(s) to infer, {len(run_dirs) - len(todo)} from cache")
    t0 = time.time()
    for i, (rid, d, dp, qp, kp, n_img) in enumerate(todo, 1):
        if model is None:
            from ultralytics import YOLO
            model = YOLO(str(weights))
        r = cal_i.loc[rid]
        H = np.array([list(x) for x in r["H"]], dtype=np.float64)
        c = geometry.Calibration(rid, H, int(r.n_calib_points), float(r.rms_reprojection_px),
                                 float(r.max_reprojection_px), [], bool(r.calibration_ok))
        det, qc = infer.predict_run(model, d, c, conf=inf["conf"], iou=inf["iou"],
                                    batch=int(inf["batch"]), device=inf["device"])
        det.to_parquet(dp, index=False)
        qc.to_parquet(qp, index=False)
        kp.write_text(json.dumps({**key, "n_images": n_img}), encoding="utf-8")
        det_parts.append(det)
        qc_parts.append(qc)
        el = time.time() - t0
        log(f"    {i}/{len(todo)} {d.name}: {n_img} frames  "
            f"({el / 60:.1f} min elapsed, ~{el / i * (len(todo) - i) / 60:.0f} min left)")
    det = pd.concat(det_parts, ignore_index=True).sort_values(["run_id", "frame_idx", "fish_slot"],
                                                               kind="stable").reset_index(drop=True)
    qc = pd.concat(qc_parts, ignore_index=True).sort_values(["run_id", "frame_idx"],
                                                            kind="stable").reset_index(drop=True)
    counts = qc.n_kept.value_counts(normalize=True).sort_index()
    log("  visible fish per frame (training-set reference in brackets): " + ", ".join(
        f"{int(k)}: {v:.1%}" + (f" [{infer.LABELLED_VISIBLE_COUNTS[int(k)]:.0%}]"
                                if int(k) in infer.LABELLED_VISIBLE_COUNTS else "")
        for k, v in counts.items()))
    log(f"  trigger fish identified in {det.groupby(['run_id', 'frame_idx']).is_trigger_fish.any().mean():.1%} of frames")
    return det, qc


def stage_features(cfg, R, E, cal, det, qc, blind_s: float) -> dict[str, pd.DataFrame]:
    window = config.window(cfg)
    gap = float(cfg["events"]["bout_gap_s"])
    B = bouts.build_bout_table(E, gaps_s=[gap], window=window, bin_s=cfg["events"]["bin_s"],
                               refractory_s=cfg["bouts"]["refractory_s"], floor_tol_s=cfg["bouts"]["floor_tol_s"])
    # The window cannot merge or split interior bouts, so bout counts overlapping
    # the window must equal the per-run in-window counts from the raw events.
    ref = R.set_index("run_id")
    for plat, col in [("active", "n_active_bouts_in_window"), ("inactive", "n_inactive_bouts_in_window")]:
        mine = B[(B.platform == plat) & B.overlaps_window].groupby("run_id").size()
        got = ref.index.map(lambda r: mine.get(r, 0))
        assert (np.asarray(got) == ref[col].to_numpy()).all(), f"{plat} bout counts disagree with the event summary"

    X = bouts.build_exposure_table(E, R, blind_s_per_trigger=blind_s,
                                   window=window, bin_s=cfg["events"]["bin_s"])
    assert (X.observed_time_s >= 0).all()
    e = cfg["effort"]
    idx, status = bouts.effort_subsample_index(B, k=int(e["k"]), b=int(e["b"]), seed=int(e["seed"]), gap_s=gap)
    below = status[~status.effort_equalized]
    log(f"  effort: {int(status.effort_equalized.sum())}/{len(status)} runs have >= {e['k']} in-window active bouts"
        + (f"; below (no pose features): {', '.join(below.run_id)}" if len(below) else ""))
    floor_s = cfg["bouts"]["refractory_s"] + cfg["bouts"]["floor_tol_s"]
    A = features.tier_a(B, E, X, R, gap_s=gap, floor_s=floor_s)
    TM = features.timing_features(R, E, B, X, A, window=window, gap_s=gap,
                                  late_min_obs_s=cfg["features"]["late_min_obs_s"])
    TC = features.tier_c(R, cal, qc, det, B, idx, status, gap_s=gap)
    return {"bouts": B, "exposure": X, "effort_index": idx, "effort_status": status,
            "features_tier_a": A, "features_timing": TM, "features_tier_c": TC}


# --------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, type=Path, help="folder holding one subfolder per run")
    ap.add_argument("--metadata", type=Path, default=None,
                    help="metadata CSV (default: metadata.csv inside --runs)")
    ap.add_argument("--out", type=Path, default=Path("out"), help="output folder (default: ./out)")
    ap.add_argument("--params", type=Path, default=None,
                    help="params YAML (default: params.yaml next to this script, if present)")
    ap.add_argument("--stop-after", choices=STAGES, default="analysis",
                    help="stop after this stage (e.g. 'check' to validate metadata only)")
    ap.add_argument("--force", choices=["calibration", "pose", "all"], default=None,
                    help="recompute cached per-run calibration and/or pose results")
    args = ap.parse_args(argv)

    t0 = time.time()
    params = args.params or (paths.PROJECT_ROOT / "params.yaml")
    cfg = config.load(params if params.exists() else None)
    project = paths.Project(args.runs.resolve(), args.out.resolve())
    metadata_csv = (args.metadata or (project.runs_dir / "metadata.csv")).resolve()
    project.ensure_dirs()
    warns: list[str] = []
    info = {"runs_dir": str(project.runs_dir), "metadata": str(metadata_csv), "out_dir": str(project.out_dir),
            "params": str(params) if params.exists() else None, "config_hash": config.hash_config(cfg),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def done(stage: str) -> bool:
        return STAGES.index(stage) >= STAGES.index(args.stop_after)

    log("[1/6] check")
    try:
        runs, run_dirs, w = stage_check(project, metadata_csv)
    except metadata.MetadataError as exc:
        raise SystemExit(f"metadata problem: {exc}")
    warns += w
    log(f"  {len(run_dirs)} run(s) to analyse ({int(runs.is_vehicle.sum())} vehicle, "
        f"{runs.compound[~runs.is_vehicle].nunique()} compounds, {runs.date.nunique()} days)")
    info["n_runs"] = len(run_dirs)
    if done("check"):
        return finish(project, info, warns, t0)

    log("[2/6] events")
    R, E, blind = stage_events(project, cfg, runs, run_dirs, warns)
    R.to_parquet(project.out_dir / "runs.parquet", index=False)
    E.to_parquet(project.out_dir / "events.parquet", index=False)
    info["blind_time"] = blind
    if done("events"):
        return finish(project, info, warns, t0)

    log("[3/6] calibration")
    cal = stage_calibration(project, cfg, run_dirs, args.force in ("calibration", "all"), warns)
    cal.to_parquet(project.out_dir / "calibration.parquet", index=False)
    if done("calibration"):
        return finish(project, info, warns, t0)

    log("[4/6] pose")
    det, qc = stage_pose(project, cfg, run_dirs, cal, args.force in ("pose", "all"))
    det.to_parquet(project.out_dir / "pose_detections.parquet", index=False)
    qc.to_parquet(project.out_dir / "pose_frame_qc.parquet", index=False)
    if done("pose"):
        return finish(project, info, warns, t0)

    log("[5/6] features")
    tables = stage_features(cfg, R, E, cal, det, qc, blind["s_per_trigger"])
    for name, df in tables.items():
        df.to_parquet(project.out_dir / f"{name}.parquet", index=False)
    if done("features"):
        return finish(project, info, warns, t0)

    log("[6/6] analysis")
    res = analysis.run(project.out_dir, cfg, log=log)
    info["results"] = res
    log(f"\nBIC selects K={res['selected_K']} ({res['selected_covariance']}), "
        f"ΔBIC {res['delta_bic']:.1f}, p_null {res['p_null']:.3f}; described partition sizes "
        f"{res['cluster_sizes']}" + (" -- descriptive only" if res["descriptive_only"] else ""))
    log(f"report: {project.analysis_dir / 'REPORT.md'}")
    finish(project, info, warns, t0)


def finish(project: paths.Project, info: dict, warns: list[str], t0: float) -> None:
    info["warnings"] = warns
    info["runtime_s"] = round(time.time() - t0, 1)
    (project.out_dir / "pipeline_run.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    if warns:
        log(f"\n{len(warns)} warning(s):")
        for w in warns:
            log(f"  - {w}")
    log(f"\ndone in {info['runtime_s'] / 60:.1f} min; outputs in {project.out_dir}")


if __name__ == "__main__":
    main()
