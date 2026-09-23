"""Pose inference with the N=5 constraint, per-frame QC, and the output table.

**Five fish are always in the tank, but not always visible.** Five is therefore
a *cap*, not a quota: detections are filtered by confidence first and only then
truncated to the top five. Forcing exactly five would fabricate one or two fish
in every frame where the shoal overlaps or a fish sits under the bright caustic
band. In the hand-labelled training set of best.pt the visible count is 5 in 79% of frames, 4 in
18% and 3 in 4% -- never fewer, never more -- and that distribution is the
reference to compare inference against.

What the cap does buy is a ranking problem instead of an open-ended one, plus a
record of how comfortable the choice was. Those comfort measures are the point:
if detection quality tracks treatment (and it will, because sedated fish sit
still and blend into the background model), that has to be visible as a
covariate before anything is interpreted.

**``fish_slot`` is an index, not an identity.** Frames are seconds apart and
non-contiguous, so no fish can be tracked between them. Every feature built on
this table must be permutation-invariant over the five fish: counts, means,
dispersions, order statistics -- never "fish 3's trajectory". The one legitimate
per-subject split is the **trigger fish**: in each active-trigger frame the fish
whose centroid falls inside the yellow platform ROI is the one that caused the
dose, which gives one labelled subject and a meaningful "trigger fish vs. the
other four" contrast. ``is_trigger_fish`` marks it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import events, geometry, naming, paths

#: Fish physically in the tank. Used as a ceiling on detections per frame, not
#: a target -- see the module docstring.
N_FISH = 5

#: Visible-fish distribution in the hand-labelled set, for QC comparison.
LABELLED_VISIBLE_COUNTS = {3: 0.04, 4: 0.18, 5: 0.79}

OUTPUT_COLUMNS = [
    "run_id", "frame_idx", "t_seconds", "fish_slot",
    "x_tank", "y_tank", "head_x", "head_y", "tail_x", "tail_y",
    "heading_rad", "bbox_area", "conf", "is_trigger_fish",
]


def _pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    """IoU between every pair of xyxy boxes."""
    if len(boxes) < 2:
        return np.zeros((len(boxes), len(boxes)))
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area[:, None] + area[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def frame_qc(boxes: np.ndarray, confs: np.ndarray, kept: np.ndarray) -> dict:
    """Per-frame detection-quality record.

    ``max_pairwise_iou`` high means one fish was probably split in two;
    ``min_pairwise_dist`` small means two detections landed on one animal;
    ``conf_5th`` is how confident the model was about the fish it was least sure
    of -- the marginal call that the N=5 rule forced.
    """
    k = boxes[kept]
    c = confs[kept]
    centres = np.column_stack([(k[:, 0] + k[:, 2]) / 2, (k[:, 1] + k[:, 3]) / 2]) if len(k) else np.zeros((0, 2))
    if len(centres) >= 2:
        d = np.hypot(centres[:, None, 0] - centres[None, :, 0],
                     centres[:, None, 1] - centres[None, :, 1])
        iu = np.triu_indices(len(centres), k=1)
        min_dist = float(d[iu].min())
        max_iou = float(_pairwise_iou(k)[iu].max())
    else:
        min_dist, max_iou = float("nan"), float("nan")
    return {
        "n_detected": int(len(confs)),
        "n_above_thresh": int((confs > 0.5).sum()),
        "n_kept": int(len(k)),
        "conf_5th": float(c[-1]) if len(c) >= N_FISH else float("nan"),
        "conf_min_kept": float(c.min()) if len(c) else float("nan"),
        "max_pairwise_iou": max_iou,
        "min_pairwise_dist": min_dist,
    }


def _heading(head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """Heading in radians, tail -> head, in the coordinate frame supplied."""
    return np.arctan2(head[:, 1] - tail[:, 1], head[:, 0] - tail[:, 0])


def predict_run(
    model,
    run_dir: Path,
    calibration: geometry.Calibration,
    *,
    conf: float = 0.25,
    iou: float = 0.65,
    batch: int = 16,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the pose model over one run. Returns ``(detections, frame_qc)``.

    Coordinates are emitted in canonical tank units via the calibration
    homography, so positions are comparable across runs even though the camera
    moves between sessions.
    """
    ev = events.read_events(run_dir, "active")
    t_by_frame = dict(zip(ev.frame_idx, ev.t_seconds))
    params = geometry.read_params(run_dir)
    roi = params["active_roi"]
    run_id = paths.run_stamp(run_dir)

    imgs = paths.trigger_images(run_dir)
    det_rows, qc_rows = [], []

    for start in range(0, len(imgs), batch):
        chunk = imgs[start : start + batch]
        results = model.predict(
            [str(p) for p in chunk], conf=conf, iou=iou, verbose=False, device=device
        )
        for img_path, res in zip(chunk, results):
            parsed = naming.parse_trigger_image(img_path.name)
            if parsed is None:
                continue
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
            confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros(0)
            kps = (
                res.keypoints.xy.cpu().numpy()
                if res.keypoints is not None
                else np.zeros((len(boxes), 3, 2))
            )

            order = np.argsort(-confs)
            kept = order[:N_FISH]
            qc = frame_qc(boxes, confs, kept)
            qc.update({"run_id": run_id, "frame_idx": parsed.frame_idx,
                       "t_seconds": t_by_frame.get(parsed.frame_idx, np.nan)})
            qc_rows.append(qc)

            if len(kept) == 0:
                continue
            b, k = boxes[kept], kps[kept]
            centre_px = np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2])

            # Trigger fish: the one inside the yellow platform ROI. Decided in
            # pixel space, where the ROI is defined.
            inside = (
                (centre_px[:, 0] >= roi["x"]) & (centre_px[:, 0] < roi["x"] + roi["w"])
                & (centre_px[:, 1] >= roi["y"]) & (centre_px[:, 1] < roi["y"] + roi["h"])
            )
            trigger_slot = -1
            if inside.any():
                # If several are inside, the most confident is the trigger fish.
                trigger_slot = int(np.flatnonzero(inside)[0])

            centre_t = calibration.to_canonical(centre_px.astype(np.float32))
            head_t = calibration.to_canonical(k[:, 0, :].astype(np.float32))
            tail_t = calibration.to_canonical(k[:, 2, :].astype(np.float32))
            heading = _heading(head_t, tail_t)

            for slot in range(len(kept)):
                det_rows.append(
                    {
                        "run_id": run_id,
                        "frame_idx": parsed.frame_idx,
                        "t_seconds": t_by_frame.get(parsed.frame_idx, np.nan),
                        "fish_slot": slot,
                        "x_tank": float(centre_t[slot, 0]),
                        "y_tank": float(centre_t[slot, 1]),
                        "head_x": float(head_t[slot, 0]),
                        "head_y": float(head_t[slot, 1]),
                        "tail_x": float(tail_t[slot, 0]),
                        "tail_y": float(tail_t[slot, 1]),
                        "heading_rad": float(heading[slot]),
                        "bbox_area": float((b[slot, 2] - b[slot, 0]) * (b[slot, 3] - b[slot, 1])),
                        "conf": float(confs[kept][slot]),
                        "is_trigger_fish": bool(slot == trigger_slot),
                    }
                )

    det = pd.DataFrame(det_rows, columns=OUTPUT_COLUMNS)
    return det, pd.DataFrame(qc_rows)

