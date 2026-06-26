"""
demo.py
-------
Run the full VSLAM pipeline (VO + Local Mapping + Loop Detection + PGO + Map Reuse).

Display layout (ORB-SLAM2 / Pangolin style)
-------------------------------------------
  Viewer window: Combined window showing the grayscale feed with tracked keypoints
                 on the left, and the pure OpenCV render of the Map (trajectory,
                 frustums, point cloud) on the right.

Modes
-----
  1. KITTI dataset  – pass --kitti /path/to/sequence/image_0
  2. Video file     – pass --video  path/to/video.mp4
  3. Webcam         – pass --webcam

Stage flags
-----------
  --no-local-map          disable local bundle adjustment thread
  --no-loop               disable loop detection thread
  --save-vocab PATH       save vocabulary after run
  --load-vocab PATH       load pre-built vocabulary (skips build delay)
  --save-map   PATH       save full map to .pkl after run          [Stage 5]
  --load-map   PATH       load existing map and relocalize into it  [Stage 5]
  --reloc-frames N        frames to attempt relocalization (default 30)
  --reloc-only            localize only, do not extend map
  --min-loop-gap N        frames between loop callbacks (dead zone, default 200)

Camera intrinsics
-----------------
  --fx --fy --cx --cy     full intrinsics in pixels
  --hfov                  horizontal FOV in degrees (fallback)

Keys during run
---------------
  q  quit
  s  save trajectory snapshot
  l  print current loop closure count
  r  print relocalization status
"""

import argparse
import sys
import os
import time
import collections
from pathlib import Path

import cv2
import numpy as np
import matplotlib
if os.environ.get("DISPLAY") or sys.platform == "darwin" or sys.platform == "win32":
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from vo_slam import (
    CameraModel, VOConfig, VisualOdometry, DetectorType,
    FeatureOverlay, TrajectoryPlot, plot_trajectory_static,
)
from vo_slam.local_mapping  import LocalMapper
from vo_slam.loop_detector  import LoopDetector, LoopEvent
from vo_slam.vocabulary     import VisualVocabulary
from vo_slam.pose_graph_optimizer import PoseGraphOptimizer    # Stage 4
from vo_slam.map_storage    import MapStorage
from vo_slam.relocalization import Relocalization


# ═══════════════════════════════════════════════════════════════════════ #
#  Dataset loaders                                                        #
# ═══════════════════════════════════════════════════════════════════════ #

def kitti_loader(folder: str):
    p     = Path(folder)
    # Detect if we are in a KITTI-style sequence (image_0 and image_1 sibling folders)
    is_kitti_stereo = False
    p_right = None
    if p.name == "image_0":
        p_right = p.parent / "image_1"
        if p_right.exists():
            is_kitti_stereo = True

    files_l = sorted(p.glob("*.png")) + sorted(p.glob("*.jpg"))
    if not files_l:
        raise FileNotFoundError(f"No images in {folder}")
    
    files_r = []
    if is_kitti_stereo:
        files_r = sorted(p_right.glob("*.png")) + sorted(p_right.glob("*.jpg"))

    for i in range(len(files_l)):
        img_l = cv2.imread(str(files_l[i]))
        img_r = None
        if is_kitti_stereo and i < len(files_r):
            img_r = cv2.imread(str(files_r[i]))
            
        if img_l is not None:
            yield img_l, img_r, i


def video_loader(path: str):
    cap = cv2.VideoCapture(path)
    fid = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame, None, fid
        fid += 1
    cap.release()


def webcam_loader(device: int = 0):
    cap = cv2.VideoCapture(device)
    fid = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame, None, fid
        fid += 1
    cap.release()


# ═══════════════════════════════════════════════════════════════════════ #
#  Blur detection                                                         #
# ═══════════════════════════════════════════════════════════════════════ #

def is_blurry(frame: np.ndarray, threshold: float = 80.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


# ═══════════════════════════════════════════════════════════════════════ #
#  Map Viewer panel — pure OpenCV, ORB-SLAM2 Pangolin style               #
# ═══════════════════════════════════════════════════════════════════════ #

_FRUSTUM_PTS = np.array([
    [ 0,  0],   # apex (camera position)
    [-6, -9],   # left wing
    [ 0, -6],   # notch
    [ 6, -9],   # right wing
], dtype=np.float32)


def _world_to_map_px(
    xyz          : np.ndarray,   # (N, 3) world coords
    cx           : float,        # canvas centre x
    cz           : float,        # canvas centre z
    scale        : float,        # pixels per metre
) -> np.ndarray:
    """Project world XZ onto the top-down canvas (Y ignored)."""
    px = (cx + xyz[:, 0] * scale).astype(np.int32)
    pz = (cz - xyz[:, 2] * scale).astype(np.int32)
    return np.stack([px, pz], axis=1)


def _draw_frustum(
    canvas   : np.ndarray,
    T_wc     : np.ndarray,      # 4×4 T_world_cam
    cx       : float,
    cz       : float,
    scale    : float,
    color    : tuple,
    size     : float = 1.0,
):
    """
    Draw a camera frustum (chevron) at the keyframe position.
    Heading is derived from the forward (+Z_cam) direction projected onto XZ.
    """
    pos   = T_wc[:3, 3]
    R     = T_wc[:3, :3]
    fwd   = R @ np.array([0.0, 0.0, 1.0])   # forward in world
    right = R @ np.array([1.0, 0.0, 0.0])   # right  in world

    # Build 2-D rotation from world XZ heading
    dx, dz = fwd[0], fwd[2]
    ang    = np.arctan2(dx, dz)              # angle in XZ plane
    cos_a, sin_a = np.cos(ang), np.sin(ang)

    pts = _FRUSTUM_PTS * size
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = (rot @ pts.T).T

    # Translate to canvas pixels
    ox = int(cx + pos[0] * scale)
    oz = int(cz - pos[2] * scale)
    screen = pts + np.array([ox, oz])
    screen = screen.astype(np.int32)

    cv2.polylines(canvas, [screen.reshape(-1, 1, 2)], True, color, 1, cv2.LINE_AA)


def render_map_viewer(
    map_points   : list,
    keyframes    : list,
    trajectory   : np.ndarray,
    loop_events  : list,
    panel_w      : int,
    panel_h      : int,
    vo_state     : str,
) -> np.ndarray:
    """
    Pure-OpenCV top-down map viewer mimicking Pangolin / ORB-SLAM2.
    """
    BG    = (10, 10, 12)
    canvas = np.full((panel_h, panel_w, 3), BG, dtype=np.uint8)

    # ── Auto-scale: fit the trajectory spread into the panel ─────────── #
    scale = 20.0                  # pixels per metre (default)
    if len(trajectory) > 0:
        # Find the bounding box of the entire trajectory
        min_x, max_x = float(np.min(trajectory[:, 0])), float(np.max(trajectory[:, 0]))
        min_z, max_z = float(np.min(trajectory[:, 2])), float(np.max(trajectory[:, 2]))
        
        span_x = max_x - min_x
        span_z = max_z - min_z
        span   = max(span_x, span_z, 1.0)
        
        # Use 80% of the smaller panel dimension to leave a margin
        scale  = min(panel_w, panel_h) * 0.80 / span
        
        # Lower the minimum clip limit (0.01) so large trajectories can fully zoom out
        scale  = float(np.clip(scale, 0.01, 120.0))

        # Center map onto the midpoint of the entire trajectory
        mid_x = (min_x + max_x) / 2.0
        mid_z = (min_z + max_z) / 2.0
    else:
        mid_x, mid_z = 0.0, 0.0

    # Adjust base offsets dynamically so the entire trajectory is centered
    cx = panel_w / 2.0 - mid_x * scale
    cz = panel_h / 2.0 + mid_z * scale

    # ── Map point cloud ───────────────────────────────────────────────── #
    if map_points:
        mp_xyz = np.array([mp.xyz for mp in map_points], dtype=np.float32)
        # Thin out if too many (draw every N-th)
        step = max(1, len(mp_xyz) // 8000)
        mp_xyz = mp_xyz[::step]
        pxs = _world_to_map_px(mp_xyz, cx, cz, scale)
        H, W = canvas.shape[:2]
        mask = (pxs[:, 0] >= 0) & (pxs[:, 0] < W) & (pxs[:, 1] >= 0) & (pxs[:, 1] < H)
        for px in pxs[mask]:
            cv2.circle(canvas, tuple(px), 1, (30, 30, 200), -1)   # red dots

    # ── Full trajectory spine (grey connecting line) ─────────────────── #
    if len(trajectory) > 1:
        traj_pxs = []
        for pos in trajectory:
            px  = int(cx + pos[0] * scale)
            pz  = int(cz - pos[2] * scale)
            traj_pxs.append([px, pz])
        traj_pxs = np.array(traj_pxs, dtype=np.int32)
        cv2.polylines(canvas, [traj_pxs.reshape(-1, 1, 2)], False,
                      (55, 55, 55), 1, cv2.LINE_AA)

    # ── Loop closure lines (cyan) ─────────────────────────────────────── #
    if loop_events and len(keyframes) > 1:
        kf_id_map = {kf.kf_id: kf for kf in keyframes}
        for ev in loop_events[-20:]:
            q = kf_id_map.get(ev["query_kf_id"])
            m = kf_id_map.get(ev["match_kf_id"])
            if q is None or m is None:
                continue
            p1 = q.T_world_cam[:3, 3]
            p2 = m.T_world_cam[:3, 3]
            u1, v1 = int(cx + p1[0]*scale), int(cz - p1[2]*scale)
            u2, v2 = int(cx + p2[0]*scale), int(cz - p2[2]*scale)
            cv2.line(canvas, (u1, v1), (u2, v2), (200, 200, 0), 1, cv2.LINE_AA)

    # ── Keyframe frustum chevrons ─────────────────────────────────────── #
    FRUSTUM_COLOR      = (0, 200,  60)    # green
    FRUSTUM_COLOR_CURR = (0, 255, 100)    # brighter for current
    for i, kf in enumerate(keyframes):
        is_last = (i == len(keyframes) - 1)
        color   = FRUSTUM_COLOR_CURR if is_last else FRUSTUM_COLOR
        size    = 1.5 if is_last else 0.9
        _draw_frustum(canvas, kf.T_world_cam, cx, cz, scale, color, size)

    # ── Title strip ───────────────────────────────────────────────────── #
    state_col = (0, 200, 60) if vo_state == "OK" else (0, 60, 220)
    cv2.putText(canvas, "Map Viewer", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)
    kf_n  = len(keyframes)
    mp_n  = len(map_points)
    lp_n  = len(loop_events)
    info  = f"KF:{kf_n}  MP:{mp_n}  LP:{lp_n}"
    tw, _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(canvas, info, (panel_w - tw[0] - 8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1, cv2.LINE_AA)

    return canvas


# ═══════════════════════════════════════════════════════════════════════ #
#  Frame Viewer panel — grayscale feed + ORB-SLAM2-style feature markers  #
# ═══════════════════════════════════════════════════════════════════════ #

def render_frame_viewer(
    frame        : np.ndarray,   # BGR or gray source frame
    cur_feats,                   # FrameFeatures (pts2d)
    vo_state     : str,
    num_kf       : int,
    num_lm       : int,          # landmark / map-point count
    num_kp       : int,          # keypoints in current frame
    process_ms   : float,
    is_kf        : bool,
    panel_w      : int,
    panel_h      : int,
) -> np.ndarray:
    """
    Frame viewer mimicking ORB-SLAM2's right-hand Pangolin panel.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    out  = cv2.cvtColor(gray,  cv2.COLOR_GRAY2BGR)
    out  = cv2.resize(out, (panel_w, panel_h))

    H, W = out.shape[:2]

    # Scale factor from original frame to panel
    orig_h, orig_w = frame.shape[:2]
    sx = panel_w / orig_w
    sy = panel_h / orig_h

    # ── Yellow hollow-square bracket markers ──────────────────────────── #
    if cur_feats is not None and len(cur_feats) > 0:
        BRACKET_COLOR  = (0, 220, 220)   # yellow (BGR)
        BRACKET_HALF   = 6               # half-size of square in pixels
        BRACKET_ARM    = 3               # length of corner arm
        THICKNESS      = 1

        for pt in cur_feats.pts2d:
            u = int(pt[0] * sx)
            v = int(pt[1] * sy)
            if not (BRACKET_HALF <= u < W - BRACKET_HALF and
                    BRACKET_HALF <= v < H - BRACKET_HALF):
                continue
            x0, y0 = u - BRACKET_HALF, v - BRACKET_HALF
            x1, y1 = u + BRACKET_HALF, v + BRACKET_HALF

            # Top-left corner
            cv2.line(out, (x0, y0), (x0 + BRACKET_ARM, y0), BRACKET_COLOR, THICKNESS)
            cv2.line(out, (x0, y0), (x0, y0 + BRACKET_ARM), BRACKET_COLOR, THICKNESS)
            # Top-right corner
            cv2.line(out, (x1, y0), (x1 - BRACKET_ARM, y0), BRACKET_COLOR, THICKNESS)
            cv2.line(out, (x1, y0), (x1, y0 + BRACKET_ARM), BRACKET_COLOR, THICKNESS)
            # Bottom-left corner
            cv2.line(out, (x0, y1), (x0 + BRACKET_ARM, y1), BRACKET_COLOR, THICKNESS)
            cv2.line(out, (x0, y1), (x0, y1 - BRACKET_ARM), BRACKET_COLOR, THICKNESS)
            # Bottom-right corner
            cv2.line(out, (x1, y1), (x1 - BRACKET_ARM, y1), BRACKET_COLOR, THICKNESS)
            cv2.line(out, (x1, y1), (x1 - BRACKET_ARM, y1), BRACKET_COLOR, THICKNESS)

    # ── Status bar — single line at the very bottom ───────────────────── #
    BAR_H     = 28
    bar_top   = H - BAR_H
    cv2.rectangle(out, (0, bar_top), (W, H), (18, 18, 18), -1)

    state_label = "MAPPING" if vo_state == "OK" else vo_state
    state_color = (0, 210, 60) if vo_state == "OK" else (40, 40, 220)

    cv2.putText(out, state_label, (8, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, state_color, 1, cv2.LINE_AA)

    sep_x = 8 + cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0] + 6
    cv2.putText(out, "|", (sep_x, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 80, 80), 1, cv2.LINE_AA)

    stats_txt = (f"  KF: {num_kf},  LM: {num_lm},  "
                 f"KP: {num_kp},  track time: {process_ms:.0f}ms")
    cv2.putText(out, stats_txt, (sep_x + 12, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)

    if is_kf:
        badge_txt = "NEW KF"
        tw, th = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
        bx = W - tw - 10
        cv2.rectangle(out, (bx - 4, 4), (W - 4, th + 8), (0, 160, 160), -1)
        cv2.putText(out, badge_txt, (bx, th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

    return out


# ═══════════════════════════════════════════════════════════════════════ #
#  Main runner                                                            #
# ═══════════════════════════════════════════════════════════════════════ #

def run(args):

    # ── Camera ───────────────────────────────────────────────────────── #
    if args.kitti:
        # Try to load calib.txt from parent folder
        p_kitti = Path(args.kitti)
        calib_file = p_kitti.parent / "calib.txt"
        if calib_file.exists():
            try:
                from run_kitti import load_calib
                K, baseline = load_calib(calib_file)
                camera = CameraModel.from_matrix(K, width=1241, height=376, baseline=baseline)
                print(f"Camera: KITTI calib found  {camera}")
            except Exception as e:
                print(f"[WARNING] Failed to parse KITTI calib: {e}")
                camera = CameraModel.kitti()
        else:
            camera = CameraModel.kitti()
            print(f"Camera: KITTI preset (seq 00)  {camera}")

    elif args.webcam or args.video:
        cap = cv2.VideoCapture(0 if args.webcam else args.video)
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if args.fx and args.fy and args.cx and args.cy:
            K      = np.array([[args.fx, 0, args.cx],
                                [0, args.fy, args.cy],
                                [0, 0, 1.0]])
            camera = CameraModel.from_matrix(K, width=W, height=H)
            print(f"Camera: intrinsics provided  {camera}")
        elif args.hfov:
            camera = CameraModel.from_fov(args.hfov, W, H)
            print(f"Camera: hfov={args.hfov}°  {camera}")
        else:
            camera = CameraModel.from_fov(60.0, W, H)
            print(f"[WARNING] No intrinsics — using hfov=60° estimate.")
            print(f"          Pass --fx --fy --cx --cy or --hfov for accuracy.")
    else:
        camera = CameraModel.from_fov(70.0, width=1241, height=376)

    print(f"K =\n{camera.K}\n")

    # ── Window configurations ─────────────────────────────────────────── #
    # Map Viewer dimension requirements
    MAP_W = 800
    MAP_H = 800

    # Frame Viewer sizes uniformly scaled according to source resolution to prevent distortion
    FRAME_H = max(480, camera.height)
    scale_f = FRAME_H / camera.height
    FRAME_W = int(camera.width * scale_f)

    # ── Stage 5: Load existing map ────────────────────────────────────── #
    saved_map       = None
    relocalizer     = None
    reloc_status    = "none"
    reloc_done      = True
    reloc_attempts  = 0
    reloc_successes = 0

    if args.load_map and Path(args.load_map).exists():
        saved_map = MapStorage.load(args.load_map)
        relocalizer = Relocalization(
            saved_map   = saved_map,
            camera_K    = camera.K,
            min_matches = 15,
            min_inliers = 12,
        )
        reloc_done   = False
        reloc_status = f"searching ({args.reloc_frames} frames)"
        print(f"Map loaded     : {saved_map.n_keyframes} KFs, "
              f"{saved_map.n_map_points} MPs")
        print(f"Relocalization : armed for first {args.reloc_frames} frames")
    elif args.load_map:
        print(f"[WARNING] Map file not found: {args.load_map}")

    # ── VO ────────────────────────────────────────────────────────────── #
    cfg = VOConfig(
        detector_type   = DetectorType.ORB,
        max_features    = 1500,
        ratio_thresh    = 0.75,
        min_inliers     = 15,
        scale_mode      = "fixed",
        fixed_scale     = 1.0,
        store_images    = False,
        kf_min_parallax = 2.0,
        kf_min_frames   = 3,
        kf_max_frames   = 15,
    )
    vo = VisualOdometry(camera, cfg)

    # ── Stage 4: Pose Graph Optimizer ────────────────────────────────── #
    pgo = PoseGraphOptimizer(vo, n_iters=20, verbose=False)
    print(f"PGO backend    : {'g2o' if pgo.__class__.__module__ else 'scipy'}")

    # ── Stage 2: Local Mapper ─────────────────────────────────────────── #
    mapper = None
    if not args.no_local_map:
        mapper = LocalMapper(
            vo             = vo,
            verbose        = False,
        )
        mapper.start()
        print("LocalMapper    : ON")
    else:
        print("LocalMapper    : OFF  (--no-local-map)")

    # ── Stage 3: Loop Detector ────────────────────────────────────────── #
    loop_events   = []
    loop_detector = None

    def on_loop(event: LoopEvent):
        loop_events.append({
            "query_kf_id" : event.query_kf_id,
            "match_kf_id" : event.match_kf_id,
            "bow_score"   : round(float(event.bow_score), 4),
            "geo_inliers" : int(event.geo_inliers),
        })
        print(f"\n  *** LOOP CLOSURE *** "
              f"KF{event.query_kf_id} <-> KF{event.match_kf_id}  "
              f"geo_inliers={event.geo_inliers}  bow={event.bow_score:.4f}\n")
        
        # ── Stage 4: trigger PGO immediately ─────────────────────────── #
        pgo.on_loop(event)

    if not args.no_loop:
        vocab = None
        if saved_map and saved_map.recognizer:
            vocab = saved_map.recognizer.vocab
            print("Vocabulary     : loaded from saved map")
        elif args.load_vocab and Path(args.load_vocab).exists():
            vocab = VisualVocabulary.load(args.load_vocab)
            print(f"Vocabulary     : loaded from {args.load_vocab}")

        loop_detector = LoopDetector(
            vocab               = vocab,
            camera_K            = camera.K,
            on_loop_detected    = on_loop,
            min_bow_score       = 0.012,
            min_geo_inliers     = 25,
            consistency         = 3,
            temporal_window     = 20,
            vocab_build_at      = 50,
            verbose             = True,
        )
        print(f"LoopDetector   : ON (min_loop_gap={args.min_loop_gap} frames)")

        if saved_map and saved_map.recognizer:
            loop_detector._recognizer = saved_map.recognizer
            print(f"BoW DB         : pre-loaded  "
                  f"({len(saved_map.recognizer.db)} entries from saved map)")

        loop_detector.start()
    else:
        print("LoopDetector   : OFF  (--no-loop)")

    # ── Wire hooks ────────────────────────────────────────────────────── #
    def on_new_keyframe(kf):
        if mapper and not args.reloc_only:
            mapper.enqueue(kf)
        if loop_detector:
            loop_detector.enqueue(kf)

    vo.on_new_keyframe = on_new_keyframe

    # ── Source ────────────────────────────────────────────────────────── #
    if args.kitti:
        source = kitti_loader(args.kitti)
    elif args.video:
        source = video_loader(args.video)
    elif args.webcam:
        source = webcam_loader()
    else:
        print("Error: No source specified.")
        print("Please provide one of the following arguments:")
        print("  --kitti <path> : Path to KITTI image_0 folder")
        print("  --video <path> : Path to video file")
        print("  --webcam       : Use webcam (device 0)")
        sys.exit(1)
    # ── Visualisation state ───────────────────────────────────────────── #
    show_gui     = not args.no_gui
    blurry_count = 0
    
    # Cache map panel — only rebuild every 3 frames
    map_cache       = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
    map_last_frame  = -999

    print(f"\nRunning. Keys: q=quit  s=snapshot  l=loops  r=reloc status\n")

    # ── Main loop ─────────────────────────────────────────────────────── #
    for img_l, img_r, fid in source:

        # ── Blur check ────────────────────────────────────────────────── #
        if is_blurry(img_l, threshold=args.blur_threshold):
            blurry_count += 1
            if show_gui:
                cv2.waitKey(1)
            continue

        # ── Stage 5: Relocalization attempt ───────────────────────────── #
        if not reloc_done and relocalizer is not None:
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            orb_r  = cv2.ORB_create(nfeatures=1000)
            kps_l, descs_l = orb_r.detectAndCompute(gray_l, None)

            if descs_l is not None and len(descs_l) > 20:
                pts_l  = np.array([kp.pt for kp in kps_l], dtype=np.float32)
                result = relocalizer.relocalize(descs_l, pts_l)

                if result.success:
                    vo.T_world_cam = result.T_world_cam.copy()
                    reloc_status   = f"OK  KF{result.matched_kf_id}  ({result.inliers} inliers)"
                    reloc_successes += 1
                    reloc_done      = True
                    print(f"  [Reloc] Recovered at frame {fid} → "
                          f"KF{result.matched_kf_id}  inliers={result.inliers}")

            reloc_attempts += 1
            if reloc_attempts >= args.reloc_frames and not reloc_done:
                reloc_status = f"failed ({reloc_attempts} attempts)"
                reloc_done   = True
                print(f"  [Reloc] Failed after {reloc_attempts} frames — starting fresh")

        # ── VO process ────────────────────────────────────────────────── #
        stats = vo.process(img_l, img_right=img_r, timestamp=fid / 30.0)
        
        # ── Build displays (GUI only) ─────────────────────────────────── #
        if show_gui:

            cur_feats = vo._last_features   # pts2d of current-frame keypoints

            # ── Map Viewer (rebuild) ──────────────────────────────────── #
            if fid % 3 == 0 or map_last_frame < 0:
                map_cache = render_map_viewer(
                    map_points  = vo.map_points,
                    keyframes   = vo.keyframes,
                    trajectory  = vo.trajectory if len(vo.trajectory) > 1
                                  else np.zeros((1, 3)),
                    loop_events = loop_events,
                    panel_w     = MAP_W,
                    panel_h     = MAP_H,
                    vo_state    = vo.state.name,
                )
                map_last_frame = fid

            # ── Frame Viewer (proportional, matches actual source aspect ratio) ── #
            frame_panel = render_frame_viewer(
                frame      = img_l,
                cur_feats  = cur_feats,
                vo_state   = vo.state.name,
                num_kf     = len(vo.keyframes),
                num_lm     = len(vo.map_points),
                num_kp     = len(cur_feats) if cur_feats else 0,
                process_ms = stats.process_ms,
                is_kf      = stats.is_keyframe,
                panel_w    = FRAME_W,
                panel_h    = FRAME_H,
            )

            # ── Combine Windows ───────────────────────────────────────── #
            # Stack the frame_panel and map_cache side-by-side
            h1, w1 = frame_panel.shape[:2]
            h2, w2 = map_cache.shape[:2]
            max_h = max(h1, h2)
            total_w = w1 + w2
            
            combined = np.zeros((max_h, total_w, 3), dtype=np.uint8)
            # Place the frame viewer on the left
            combined[:h1, :w1] = frame_panel
            # Place the map viewer on the right
            combined[:h2, w1:w1+w2] = map_cache

            # Display as a single window
            cv2.imshow("VSLAM Viewer (Frame + Map)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                _save_snapshot(vo, fid, loop_events)
            if key == ord('l'):
                print(f"  Loop closures: {len(loop_events)}")
                for ev in loop_events:
                    print(f"    KF{ev['query_kf_id']} <-> KF{ev['match_kf_id']}  "
                          f"bow={ev['bow_score']}  inliers={ev['geo_inliers']}")
            if key == ord('r'):
                print(f"  Relocalization: {reloc_status}")
                if relocalizer:
                    print(f"  {relocalizer.summary()}")

        # Console log every 10 frames
        if fid % 10 == 0:
            pos    = vo.T_world_cam[:3, 3]
            kf_tag = " [KF]"   if stats.is_keyframe     else ""
            lp_tag = f" [LOOP x{len(loop_events)}]" if loop_events else ""
            rl_tag = f" [RELOC]" if reloc_successes > 0 and fid < 50 else ""
            print(f"  Frame {fid:4d} | "
                  f"matched={stats.num_matched:4d} | "
                  f"inliers={stats.num_inliers:4d} | "
                  f"MPs={len(vo.map_points):5d} | "
                  f"pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) | "
                  f"{stats.process_ms:.1f}ms{kf_tag}{lp_tag}{rl_tag}")

    # ── Shutdown threads ──────────────────────────────────────────────── #
    if mapper:
        mapper.stop()
    if loop_detector:
        loop_detector.stop()

    # ── Save vocabulary ───────────────────────────────────────────────── #
    if args.save_vocab and loop_detector and loop_detector._recognizer:
        loop_detector._recognizer.vocab.save(args.save_vocab)
        print(f"Vocabulary saved → {args.save_vocab}")

    # ── Stage 5: Save map ─────────────────────────────────────────────── #
    if args.save_map and not args.reloc_only:
        MapStorage.save(
            path          = args.save_map,
            vo            = vo,
            loop_detector = loop_detector,
        )

    # ── Save poses (from run_kitti) ───────────────────────────────────── #
    out_path_poses = "results_demo.txt"
    with open(out_path_poses, "w") as f:
        for T in vo.pose_graph.poses:
            f.write(" ".join(f"{v:.6e}" for v in T[:3].flatten()) + "\n")

    # ── Final summary ─────────────────────────────────────────────────── #
    print("\n" + vo.summary())
    print(f"=== Run Summary ===")
    print(f"  Blurry frames skipped : {blurry_count}")
    print(f"  Loop closures         : {len(loop_events)}")
    print(f"  PGO runs              : {pgo.n_optimizations}")
    print(f"  PGO backend           : {pgo.summary()}")
    print(f"  Poses saved to        : {out_path_poses}")
    
    if mapper:
        print(f"  BA runs               : {mapper.n_ba_runs}")
        print(f"  Points culled         : {mapper.n_pts_culled}")
        print(f"  KFs culled            : {mapper.n_kfs_culled}")
    if relocalizer:
        print(f"  Relocalization        : {relocalizer.summary()}")

    # ── Loop detector summary ─────────────────────────────────────────── #
    if loop_detector:
        print(f"\n=== Loop Detector Detail ===")
        print(f"  {loop_detector.summary()}")
        print(f"  Suppressed events : {loop_detector.n_suppressed}")

    # ── Final trajectory plot ─────────────────────────────────────────── #
    if len(vo.trajectory) > 1:
        out_path = args.output or "trajectory.png"
        fig = plot_trajectory_static(
            vo.trajectory, vo.map_points, vo.keyframes,
            save_path = out_path,
            title     = (f"Visual SLAM  |  {len(vo.trajectory)} frames  |  "
                         f"{len(vo.keyframes)} KFs  |  "
                         f"{len(loop_events)} loops"),
        )
        print(f"\nTrajectory saved → {out_path}")
        if show_gui and matplotlib.get_backend() != "Agg":
            plt.show()
        plt.close(fig)

    if show_gui:
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════ #
#  Snapshot helper                                                        #
# ═══════════════════════════════════════════════════════════════════════ #

def _save_snapshot(vo, fid, loop_events=None):
    path = f"snapshot_frame{fid:04d}.png"
    if len(vo.trajectory) > 1:
        fig = plot_trajectory_static(
            vo.trajectory, vo.map_points, vo.keyframes,
            save_path = path,
            title     = f"Snapshot @ frame {fid}  |  {len(loop_events or [])} loops",
        )
        plt.close(fig)
        print(f"  → Snapshot saved: {path}")


# ═══════════════════════════════════════════════════════════════════════ #
#  CLI                                                                    #
# ═══════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VSLAM Demo — VO + Local Mapping + Loop Detection + Map Reuse"
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--kitti",  type=str, help="Path to KITTI image_0 folder")
    src.add_argument("--video",  type=str, help="Path to video file")
    src.add_argument("--webcam", action="store_true", help="Use webcam (device 0)")

    parser.add_argument("--fx",   type=float)
    parser.add_argument("--fy",   type=float)
    parser.add_argument("--cx",   type=float)
    parser.add_argument("--cy",   type=float)
    parser.add_argument("--hfov", type=float)

    parser.add_argument("--no-local-map", action="store_true")
    parser.add_argument("--no-loop",      action="store_true")
    parser.add_argument("--save-vocab",   type=str)
    parser.add_argument("--load-vocab",   type=str)
    parser.add_argument("--save-map",     type=str)
    parser.add_argument("--load-map",     type=str)
    parser.add_argument("--reloc-frames", type=int, default=30)
    parser.add_argument("--reloc-only",   action="store_true")
    parser.add_argument("--min-loop-gap", type=int, default=200, help="Frames between loop callbacks (dead zone)")
    parser.add_argument("--output",       type=str, default="trajectory.png")
    parser.add_argument("--no-gui",       action="store_true")
    parser.add_argument("--blur-threshold", type=float, default=80.0)

    args = parser.parse_args()
    run(args)