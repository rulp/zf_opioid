"""Background models, platform landmarks and per-run calibration.

Why this stage exists: the operator redraws both ROIs by hand every run, and
the camera drifts between sessions. Nothing downstream can compare a position
in one run to a position in another until both are expressed in a common
frame -- the canonical tank frame, [0, 1] x [0, 1].

What the imagery allows:

* **The tank fills 95-98% of the frame.** Its corners sit essentially on the
  image boundary, so they cannot be segmented by intensity and are not used.
* **The yellow platform can be found by colour** and is a genuine physical
  object, unlike the hand-drawn ROI rectangles, so it is the primary landmark.
* **Two dark fittings** -- top-right and bottom-left -- span the frame diagonal
  and condition the fit far better than the small platform alone.

Saved JPEGs carry burned-in overlay graphics: two lines of status text in the
top-left corner whose digits change every frame, plus static red and blue ROI
rectangles. The static rectangles vanish under background subtraction; the
changing text does not and is masked explicitly.

The pixel constants below (overlay box, colour gate, platform size, canonical
landmark positions) describe this rig and camera. A different rig or camera
position needs them re-measured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import paths

#: Burned-in status text, measured as identical across runs. Generous margin.
OVERLAY_TEXT_BOX = (0, 0, 355, 50)  # x0, y0, x1, y1

#: Frames sampled per run when building the background model.
DEFAULT_N_SAMPLES = 120

#: Yellow-platform colour gate, OpenCV HSV (hue 0-180).
YELLOW_HUE = (10, 35)
YELLOW_MIN_SAT = 70
YELLOW_MIN_VAL = 40

#: Plausible physical platform extent in pixels, from the observed median of
#: ~128 x 131 with generous slack.
PLATFORM_SIDE_PX = (95, 185)
PLATFORM_AREA_PX = (9_000, 30_000)


def overlay_mask(shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask, True where the burned-in status text is drawn."""
    m = np.zeros(shape[:2], dtype=bool)
    x0, y0, x1, y1 = OVERLAY_TEXT_BOX
    m[y0:y1, x0:x1] = True
    return m


@dataclass
class Background:
    """Per-run background model built from sampled trigger frames."""

    run_id: str
    median_bgr: np.ndarray  # (H, W, 3) uint8
    n_samples: int

    def save(self, path: Path) -> None:
        np.savez_compressed(path, median_bgr=self.median_bgr, n_samples=self.n_samples, run_id=self.run_id)

    @classmethod
    def load(cls, path: Path) -> "Background":
        z = np.load(path, allow_pickle=False)
        return cls(run_id=str(z["run_id"]), median_bgr=z["median_bgr"], n_samples=int(z["n_samples"]))


def background_cache_path(cache_dir: Path, run_id: str) -> Path:
    return Path(cache_dir) / f"bg_{run_id}.npz"


def build_background(
    run_dir: Path,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = 0,
    cache_dir: Path | None = None,
) -> Background:
    """Per-pixel median over randomly sampled trigger frames.

    Trigger frames are conditioned on a fish being over the *active* platform,
    so a fish is present there in nearly every sampled frame. The median still
    recovers the background as long as per-pixel occupancy stays below 50%,
    which it comfortably does (measured peak ~0.06 inside the active ROI).

    With ``cache_dir`` the model is saved there and reused on later calls.
    """
    run_id = paths.run_stamp(run_dir)
    cache_path = background_cache_path(cache_dir, run_id) if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        return Background.load(cache_path)

    imgs = paths.trigger_images(run_dir)
    if not imgs:
        raise ValueError(f"{run_dir} has no trigger images")
    rng = np.random.default_rng(seed)
    k = min(n_samples, len(imgs))
    sel = [imgs[i] for i in rng.choice(len(imgs), k, replace=False)]

    stack = np.stack([cv2.imread(str(p)) for p in sel])
    median_bgr = np.median(stack, axis=0).astype(np.uint8)

    bg = Background(run_id, median_bgr, k)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        bg.save(cache_path)
    return bg


def read_params(run_dir: Path) -> dict:
    with open(run_dir / paths.PARAMS_JSON) as fh:
        return json.load(fh)


def roi_centre(roi: dict) -> tuple[float, float]:
    return (roi["x"] + roi["w"] / 2.0, roi["y"] + roi["h"] / 2.0)


# --- landmark detection ----------------------------------------------------


@dataclass
class PlatformFit:
    """Detected yellow platform, as a rotated rectangle."""

    centre: tuple[float, float]
    size: tuple[float, float]
    angle: float
    corners: np.ndarray  # (4, 2) float32, ordered TL, TR, BR, BL
    area: float
    score: float
    offset_from_roi: tuple[float, float]


def detect_yellow_platform(bg: Background, params: dict) -> PlatformFit | None:
    """Locate the physical yellow platform in a run's median image.

    The operator's ``active_roi`` is used only as a weak spatial prior for
    choosing among candidate contours -- the returned geometry comes from the
    image, and ``offset_from_roi`` reports how far the hand-drawn box sits from
    the physical platform.

    Candidates are scored on plausible size, squareness and proximity to the
    prior, because plain "largest contour" merges the platform with warm-toned
    tank structure in a minority of runs.
    """
    hsv = cv2.cvtColor(grey_world(bg.median_bgr), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (
        (h >= YELLOW_HUE[0])
        & (h <= YELLOW_HUE[1])
        & (s >= YELLOW_MIN_SAT)
        & (v >= YELLOW_MIN_VAL)
    ).astype(np.uint8)
    m[overlay_mask(m.shape)] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    prior = np.array(roi_centre(params["active_roi"]))
    best: PlatformFit | None = None
    for c in contours:
        area = cv2.contourArea(c)
        if not (PLATFORM_AREA_PX[0] <= area <= PLATFORM_AREA_PX[1]):
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, hgt), angle = rect
        short, long_ = sorted((w, hgt))
        if not (PLATFORM_SIDE_PX[0] <= short and long_ <= PLATFORM_SIDE_PX[1]):
            continue
        squareness = short / long_ if long_ else 0.0
        if squareness < 0.75:
            continue
        fill = area / (w * hgt) if w * hgt else 0.0  # contour vs its bounding rect
        dist = float(np.hypot(cx - prior[0], cy - prior[1]))
        score = squareness * fill * float(np.exp(-dist / 60.0))
        if best is None or score > best.score:
            corners = _order_corners(cv2.boxPoints(rect))
            best = PlatformFit(
                centre=(float(cx), float(cy)),
                size=(float(w), float(hgt)),
                angle=float(angle),
                corners=corners.astype(np.float32),
                area=float(area),
                score=float(score),
                offset_from_roi=(float(cx - prior[0]), float(cy - prior[1])),
            )
    return best


def platform_identity_check(bg: Background, params: dict) -> dict:
    """Confirm ``active_roi`` sits on the *yellow* platform, not the white one.

    Both ROIs are redrawn by hand every run, so an active/inactive swap is
    possible and would silently invert the entire endpoint. Yellow is strongly
    saturated and white is not, which separates them cleanly (observed
    saturation ~102 vs ~36), so this check is cheap and worth running on every
    run before anything else uses the ROIs.
    """
    hsv = cv2.cvtColor(grey_world(bg.median_bgr), cv2.COLOR_BGR2HSV)
    out: dict = {}
    for name in ("active_roi", "inactive_roi"):
        r = params[name]
        sub = hsv[r["y"] : r["y"] + r["h"], r["x"] : r["x"] + r["w"]]
        out[f"{name}_hue"] = float(np.median(sub[..., 0]))
        out[f"{name}_sat"] = float(np.median(sub[..., 1]))
        out[f"{name}_val"] = float(np.median(sub[..., 2]))
    out["sat_ratio"] = out["active_roi_sat"] / max(out["inactive_roi_sat"], 1e-6)
    out["active_is_yellow"] = bool(
        out["active_roi_sat"] > out["inactive_roi_sat"]
        and YELLOW_HUE[0] <= out["active_roi_hue"] <= YELLOW_HUE[1]
    )
    return out


def detect_corner_fittings(bg: Background) -> dict[str, tuple[float, float] | None]:
    """Locate the dark fittings at the top-right and bottom-left of the tank.

    These are the only landmarks present in every run that span the frame
    diagonal; without them a homography fitted to the small platform alone
    extrapolates badly across the tank.
    """
    v = cv2.cvtColor(bg.median_bgr, cv2.COLOR_BGR2HSV)[..., 2]
    dark = (v < 45).astype(np.uint8)
    dark[overlay_mask(dark.shape)] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(dark, 8)

    cands = [
        (float(cent[i][0]), float(cent[i][1]), int(stats[i, cv2.CC_STAT_AREA]))
        for i in range(1, n)
        if 250 < stats[i, cv2.CC_STAT_AREA] < 8000
    ]
    H, W = dark.shape

    def pick(corner: tuple[float, float]) -> tuple[float, float] | None:
        near = [
            (np.hypot(x - corner[0], y - corner[1]), x, y)
            for x, y, _ in cands
            if np.hypot(x - corner[0], y - corner[1]) < 160
        ]
        if not near:
            return None
        _, x, y = min(near)
        return (x, y)

    return {
        "top_right": pick((W - 60.0, 30.0)),
        "bottom_left": pick((50.0, H - 55.0)),
    }


def grey_world(bgr: np.ndarray) -> np.ndarray:
    """Remove a global colour cast by equalising the channel means.

    A run recorded with the camera's white balance failed renders the whole
    scene orange, and the absolute hue gate then matches most of the frame.
    Normalising first keys the gate on *relative* colour, and changes
    well-exposed runs by under a pixel.
    """
    f = bgr.astype(np.float32)
    mu = f.reshape(-1, 3).mean(axis=0)
    return np.clip(f * (mu.mean() / np.maximum(mu, 1e-6)), 0, 255).astype(np.uint8)


def platform_from_roi(params: dict) -> PlatformFit:
    """Fallback platform estimate taken from the operator's hand-drawn ROI.

    Only for runs where the imagery cannot yield the physical platform. Where
    both are available the ROI centre typically sits within a few pixels of
    the platform centre, so this is a defensible substitute -- but it is
    recorded as ``platform_source='roi_fallback'`` because it is the
    operator's box, not a measurement.
    """
    r = params["active_roi"]
    cx, cy = roi_centre(r)
    x0, y0 = r["x"], r["y"]
    x1, y1 = r["x"] + r["w"], r["y"] + r["h"]
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    return PlatformFit(
        centre=(cx, cy),
        size=(float(r["w"]), float(r["h"])),
        angle=0.0,
        corners=corners,
        area=float(r["w"] * r["h"]),
        score=0.0,
        offset_from_roi=(0.0, 0.0),
    )


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float64)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]]
    )


# --- canonical frame and homography ---------------------------------------

#: Where the yellow platform sits in the canonical [0,1] x [0,1] tank frame.
#: Taken from the median observed platform geometry so the canonical frame is
#: close to the average run and the transforms stay near-identity.
CANONICAL_PLATFORM = {
    "cx": 128.0 / 640.0,
    "cy": 158.0 / 480.0,
    "w": 129.0 / 640.0,
    "h": 131.0 / 480.0,
}
CANONICAL_FITTINGS = {
    "top_right": (573.0 / 640.0, 27.0 / 480.0),
    "bottom_left": (45.0 / 640.0, 427.0 / 480.0),
}


def canonical_platform_corners() -> np.ndarray:
    c = CANONICAL_PLATFORM
    x0, x1 = c["cx"] - c["w"] / 2, c["cx"] + c["w"] / 2
    y0, y1 = c["cy"] - c["h"] / 2, c["cy"] + c["h"] / 2
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


@dataclass
class Calibration:
    """Per-run image -> canonical tank mapping, with QC."""

    run_id: str
    H: np.ndarray | None
    n_points: int
    rms_reprojection_px: float
    max_reprojection_px: float
    point_names: list[str] = field(default_factory=list)
    ok: bool = False
    note: str = ""

    def to_canonical(self, xy: np.ndarray) -> np.ndarray:
        """Map (N, 2) image pixel coordinates into the canonical frame."""
        if self.H is None:
            raise ValueError(f"run {self.run_id} has no homography")
        pts = np.asarray(xy, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)


def fit_calibration(
    run_id: str, platform: PlatformFit | None, fittings: dict, image_size: tuple[int, int]
) -> Calibration:
    """Fit the image -> canonical homography from platform corners + fittings.

    Reprojection error is reported in *pixels* (canonical residuals scaled back
    by the image size) so it is directly interpretable.
    """
    if platform is None:
        return Calibration(run_id, None, 0, float("nan"), float("nan"), [], False, "no platform")

    W, H_img = image_size
    src = [platform.corners[i] for i in range(4)]
    dst = list(canonical_platform_corners())
    names = ["platform_tl", "platform_tr", "platform_br", "platform_bl"]

    for key, canon in CANONICAL_FITTINGS.items():
        pt = fittings.get(key)
        if pt is not None:
            src.append(np.array(pt, dtype=np.float32))
            dst.append(np.array(canon, dtype=np.float32))
            names.append(key)

    src_a = np.array(src, dtype=np.float32)
    dst_a = np.array(dst, dtype=np.float32)
    Hm, _ = cv2.findHomography(src_a, dst_a, method=0)
    if Hm is None:
        return Calibration(run_id, None, len(src), float("nan"), float("nan"), names, False,
                           "findHomography failed")

    proj = cv2.perspectiveTransform(src_a.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    resid = (proj - dst_a) * np.array([W, H_img])
    dists = np.hypot(resid[:, 0], resid[:, 1])
    return Calibration(
        run_id=run_id,
        H=Hm,
        n_points=len(src),
        rms_reprojection_px=float(np.sqrt((dists**2).mean())),
        max_reprojection_px=float(dists.max()),
        point_names=names,
        ok=True,
    )


def calibrate_run(
    run_dir: Path, *, n_samples: int = DEFAULT_N_SAMPLES, cache_dir: Path | None = None
) -> dict:
    """Full per-run geometry record: landmarks, homography, QC.

    Returns a flat dict suitable for a DataFrame row. The homography itself is
    returned under ``"H"`` as a nested list so it survives a parquet round-trip.
    """
    run_id = paths.run_stamp(run_dir)
    bg = build_background(run_dir, n_samples=n_samples, cache_dir=cache_dir)
    params = read_params(run_dir)

    platform = detect_yellow_platform(bg, params)
    platform_source = "detected"
    if platform is None:
        platform = platform_from_roi(params)
        platform_source = "roi_fallback"

    fittings = detect_corner_fittings(bg)
    h_img, w_img = bg.median_bgr.shape[:2]
    cal = fit_calibration(run_id, platform, fittings, (w_img, h_img))
    ident = platform_identity_check(bg, params)

    a_c = roi_centre(params["active_roi"])
    i_c = roi_centre(params["inactive_roi"])
    row = {
        "run_id": run_id,
        "n_bg_samples": bg.n_samples,
        "platform_source": platform_source,
        "platform_cx": platform.centre[0],
        "platform_cy": platform.centre[1],
        "platform_w": platform.size[0],
        "platform_h": platform.size[1],
        "platform_angle": platform.angle,
        "roi_offset_dx": platform.offset_from_roi[0],
        "roi_offset_dy": platform.offset_from_roi[1],
        "fitting_top_right": fittings["top_right"],
        "fitting_bottom_left": fittings["bottom_left"],
        "n_calib_points": cal.n_points,
        "rms_reprojection_px": cal.rms_reprojection_px,
        "max_reprojection_px": cal.max_reprojection_px,
        "calibration_ok": cal.ok,
        "H": cal.H.tolist() if cal.H is not None else None,
        "active_roi_px_area": params["active_roi"]["w"] * params["active_roi"]["h"],
        "inactive_roi_px_area": params["inactive_roi"]["w"] * params["inactive_roi"]["h"],
    }
    row.update(ident)

    if cal.ok:
        pts = cal.to_canonical(np.array([a_c, i_c], dtype=np.float32))
        row["active_roi_canon_x"], row["active_roi_canon_y"] = map(float, pts[0])
        row["inactive_roi_canon_x"], row["inactive_roi_canon_y"] = map(float, pts[1])
    return row
