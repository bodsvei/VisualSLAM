"""
demo_rerun.py
-------------
Drop-in replacement for demo.py that streams the full VSLAM pipeline
into **Rerun** for a fully navigable, high-performance 3D viewport.

What you get in the Rerun viewer
---------------------------------
  world/points          – live 3D point cloud (millions of points, GPU-rendered)
  world/trajectory      – camera path as a 3D polyline
  world/keyframes/kf_N  – every keyframe as a camera frustum (T_world_cam)
  world/loop_closures   – cyan edges connecting loop-closed keyframes
  camera/image          – live grayscale feed with ORB feature brackets
  metrics/*             – matched / inliers / process_ms time-series plots

Usage (identical flags to demo.py)
------------------------------------
  python demo_rerun.py --kitti /data/kitti/sequences/00/image_0
  python demo_rerun.py --video path/to/video.mp4
  python demo_rerun.py --webcam

Rerun connection options
------------------------
  --rr-spawn      (default) spawn a local Rerun viewer automatically
  --rr-connect    stream to a running viewer on localhost:9876
  --rr-save FILE  write a .rrd file for offline replay
  --no-gui        suppress the OpenCV status window (Rerun viewer still runs)

Tested against rerun-sdk 0.26 (compatible with ≥ 0.23).

Install:  pip install rerun-sdk
"""

import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np

# ── Rerun ──────────────────────────────────────────────────────────────── #
try:
    import rerun as rr
    import rerun.blueprint as rrb
    _HAS_RERUN = True
except ImportError:
    print("[WARNING] rerun-sdk not installed.  Run:  pip install rerun-sdk")
    _HAS_RERUN = False

sys.path.insert(0, str(Path(__file__).parent))

from vo_slam import (
    CameraModel, VOConfig, VisualOdometry, DetectorType,
    plot_trajectory_static,
)
from vo_slam.local_mapping       import LocalMapper
from vo_slam.loop_detector       import LoopDetector, LoopEvent
from vo_slam.vocabulary          import VisualVocabulary
from vo_slam.pose_graph_optimizer import PoseGraphOptimizer
from vo_slam.map_storage         import MapStorage
from vo_slam.relocalization      import Relocalization

import matplotlib
if os.environ.get("DISPLAY") or sys.platform == "darwin" or sys.platform == "win32":
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════ #
#  Math utils                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

def _R_to_quat_xyzw(R: np.ndarray) -> list:
    """3×3 rotation matrix → quaternion as [x, y, z, w] (Rerun convention)."""
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5 / float(np.sqrt(t + 1.0))
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]))
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]))
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * float(np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]))
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


# ═══════════════════════════════════════════════════════════════════════════ #
#  Rerun logger                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

class RerunLogger:
    """
    Translates VSLAM data structures → Rerun archetypes.
    All methods are safe no-ops when Rerun is unavailable.

    API compatibility (0.23 → 0.26+):
      rr.set_time_sequence → rr.set_time("name", sequence=value)  [0.23+]
      rr.Scalar            → rr.Scalars                           [0.24+]
    """

    _COL_POINT   = np.array([60,  60, 220], dtype=np.uint8)
    _COL_TRAJ    = np.array([200, 200, 200], dtype=np.uint8)
    _COL_LOOP    = np.array([220, 200,  20], dtype=np.uint8)

    def __init__(self, camera: "CameraModel", enabled: bool = True):
        self.enabled = enabled and _HAS_RERUN
        self.camera  = camera
        self._kf_ids_logged: set = set()

        if not self.enabled:
            return

        self._img_w = camera.width
        self._img_h = camera.height
        self._fx    = float(camera.fx)
        self._fy    = float(camera.fy)
        self._cx    = float(camera.cx)
        self._cy    = float(camera.cy)

    def setup_blueprint(self):
        if not self.enabled:
            return
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="3-D Scene", origin="/world"),
                rrb.Vertical(
                    rrb.Spatial2DView(name="Camera Feed", origin="/camera/image"),
                    rrb.TimeSeriesView(name="Metrics", origin="/metrics"),
                    row_shares=[3, 1],
                ),
                column_shares=[3, 2],
            ),
            rrb.BlueprintPanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
        )
        rr.send_blueprint(blueprint)

    # ── per-frame ───────────────────────────────────────────────────────── #

    def log_frame(self, fid, frame_bgr, T_world_cam, cur_feats, stats, vo_state):
        if not self.enabled:
            return

        # rr.set_time_sequence was removed in 0.23; use rr.set_time(sequence=)
        rr.set_time("frame", sequence=fid)

        t    = T_world_cam[:3, 3]
        R    = T_world_cam[:3, :3]
        xyzw = _R_to_quat_xyzw(R)

        # Current camera pose as a Pinhole (renders a frustum in the 3D view)
        rr.log(
            "world/camera_current",
            rr.Transform3D(
                translation=t,
                rotation=rr.Quaternion(xyzw=xyzw),
            ),
        )
        rr.log(
            "world/camera_current",
            rr.Pinhole(
                focal_length=[self._fx, self._fy],
                principal_point=[self._cx, self._cy],
                width=self._img_w,
                height=self._img_h,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=1.5,   # Fix frustum depth; prevents scene-scale inflation
            ),
        )

        # Live grayscale feed with ORB bracket overlays
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        rr.log("camera/image", rr.Image(gray, color_model="L"))

        if cur_feats is not None and len(cur_feats) > 0:
            pts = np.array(cur_feats.pts2d, dtype=np.float32)
            rr.log(
                "camera/image/keypoints",
                rr.Points2D(
                    pts,
                    radii=3.0,
                    colors=np.tile([0, 220, 220], (len(pts), 1)).astype(np.uint8),
                ),
            )

        # rr.Scalar was removed in 0.24; rr.Scalars is the current API
        rr.log("metrics/matched",    rr.Scalars(float(stats.num_matched)))
        rr.log("metrics/inliers",    rr.Scalars(float(stats.num_inliers)))
        rr.log("metrics/process_ms", rr.Scalars(float(stats.process_ms)))

    # ── map update ──────────────────────────────────────────────────────── #

    def log_map(self, map_points, keyframes, trajectory, loop_events):
        if not self.enabled:
            return

        # 3-D point cloud (capped at 150 k for smooth rendering)
        if map_points:
            mp_xyz = np.array([mp.xyz for mp in map_points], dtype=np.float32)
            step   = max(1, len(mp_xyz) // 150_000)
            mp_xyz = mp_xyz[::step]
            n      = len(mp_xyz)
            rr.log(
                "world/points",
                rr.Points3D(mp_xyz, radii=0.03,
                            colors=np.tile(self._COL_POINT, (n, 1))),
            )

        # Trajectory spine
        if len(trajectory) > 1:
            rr.log(
                "world/trajectory",
                rr.LineStrips3D([trajectory], colors=[self._COL_TRAJ], radii=0.02),
            )

        # Keyframe frustums — logged incrementally (only new KFs each call)
        for kf in keyframes:
            if kf.kf_id in self._kf_ids_logged:
                continue
            self._kf_ids_logged.add(kf.kf_id)
            t    = kf.T_world_cam[:3, 3]
            R    = kf.T_world_cam[:3, :3]
            xyzw = _R_to_quat_xyzw(R)
            path = f"world/keyframes/kf_{kf.kf_id:05d}"
            rr.log(path, rr.Transform3D(
                translation=t,
                rotation=rr.Quaternion(xyzw=xyzw),
            ))
            rr.log(path, rr.Pinhole(
                focal_length=[self._fx * 0.5, self._fy * 0.5],
                principal_point=[self._cx * 0.5, self._cy * 0.5],
                width=self._img_w // 2,
                height=self._img_h // 2,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.8,   # Fix frustum depth; prevents scene-scale inflation
            ))

        # Loop closure edges
        if loop_events and len(keyframes) > 1:
            kf_map = {kf.kf_id: kf for kf in keyframes}
            strips = []
            for ev in loop_events:
                q = kf_map.get(ev["query_kf_id"])
                m = kf_map.get(ev["match_kf_id"])
                if q is None or m is None:
                    continue
                strips.append(np.array([q.T_world_cam[:3, 3],
                                        m.T_world_cam[:3, 3]], dtype=np.float32))
            if strips:
                rr.log(
                    "world/loop_closures",
                    rr.LineStrips3D(strips,
                                   colors=[self._COL_LOOP] * len(strips),
                                   radii=0.015),
                )


# ═══════════════════════════════════════════════════════════════════════════ #
#  Dataset loaders  (identical to demo.py)                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

def kitti_loader(folder: str):
    p = Path(folder)
    p_right = None
    is_stereo = False
    if p.name == "image_0":
        p_right = p.parent / "image_1"
        if p_right.exists():
            is_stereo = True
    files_l = sorted(p.glob("*.png")) + sorted(p.glob("*.jpg"))
    if not files_l:
        raise FileNotFoundError(f"No images in {folder}")
    files_r = sorted(p_right.glob("*.png")) + sorted(p_right.glob("*.jpg")) if is_stereo else []
    for i, fl in enumerate(files_l):
        img_l = cv2.imread(str(fl))
        img_r = cv2.imread(str(files_r[i])) if is_stereo and i < len(files_r) else None
        if img_l is not None:
            yield img_l, img_r, i

def video_loader(path: str):
    cap, fid = cv2.VideoCapture(path), 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        yield frame, None, fid; fid += 1
    cap.release()

def webcam_loader(device: int = 0):
    cap, fid = cv2.VideoCapture(device), 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        yield frame, None, fid; fid += 1
    cap.release()

def is_blurry(frame: np.ndarray, threshold: float = 80.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main runner                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

def run(args):

    # ── Camera ────────────────────────────────────────────────────────────
    if args.kitti:
        p_kitti    = Path(args.kitti)
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
            K      = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1.0]])
            camera = CameraModel.from_matrix(K, width=W, height=H)
        elif args.hfov:
            camera = CameraModel.from_fov(args.hfov, W, H)
        else:
            camera = CameraModel.from_fov(60.0, W, H)
            print("[WARNING] No intrinsics — using hfov=60° estimate.")
    else:
        camera = CameraModel.from_fov(70.0, width=1241, height=376)
    print(f"K =\n{camera.K}\n")

    # ── Rerun init ────────────────────────────────────────────────────────
    rr_logger = RerunLogger(camera, enabled=_HAS_RERUN)
    if _HAS_RERUN:
        rr.init("vo_slam", spawn=False)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        if args.rr_save:
            rr.save(args.rr_save)
            print(f"Rerun: saving to {args.rr_save}")
        elif args.rr_connect:
            rr.connect()
            print("Rerun: connected to viewer on localhost:9876")
        else:
            rr.spawn()
            print("Rerun: viewer spawned.  Navigate freely in the 3-D panel.")

        rr_logger.setup_blueprint()

    # ── Stage 5: Load map ─────────────────────────────────────────────────
    saved_map = relocalizer = None
    reloc_status   = "none"
    reloc_done     = True
    reloc_attempts = reloc_successes = 0

    if args.load_map and Path(args.load_map).exists():
        saved_map = MapStorage.load(args.load_map)
        relocalizer = Relocalization(
            saved_map=saved_map, camera_K=camera.K,
            min_matches=15, min_inliers=12,
        )
        reloc_done   = False
        reloc_status = f"searching ({args.reloc_frames} frames)"
        print(f"Map loaded     : {saved_map.n_keyframes} KFs, {saved_map.n_map_points} MPs")

    # ── VO + PGO ──────────────────────────────────────────────────────────
    cfg = VOConfig(
        detector_type=DetectorType.ORB, max_features=1500,
        ratio_thresh=0.75, min_inliers=15, scale_mode="fixed",
        fixed_scale=1.0, store_images=False,
        kf_min_parallax=2.0, kf_min_frames=3, kf_max_frames=15,
    )
    vo  = VisualOdometry(camera, cfg)
    pgo = PoseGraphOptimizer(vo, n_iters=20, verbose=False)

    # ── Local mapper ──────────────────────────────────────────────────────
    mapper = None
    if not args.no_local_map:
        mapper = LocalMapper(vo=vo, verbose=False)
        mapper.start()
        print("LocalMapper    : ON")

    # ── Loop detector ─────────────────────────────────────────────────────
    loop_events   = []
    loop_detector = None

    def on_loop(event: LoopEvent):
        loop_events.append({
            "query_kf_id": event.query_kf_id,
            "match_kf_id": event.match_kf_id,
            "bow_score"  : round(float(event.bow_score), 4),
            "geo_inliers": int(event.geo_inliers),
        })
        print(f"\n  *** LOOP *** KF{event.query_kf_id} <-> KF{event.match_kf_id} "
              f"inliers={event.geo_inliers}\n")
        pgo.on_loop(event)

    if not args.no_loop:
        vocab = None
        if saved_map and saved_map.recognizer:
            vocab = saved_map.recognizer.vocab
        elif args.load_vocab and Path(args.load_vocab).exists():
            vocab = VisualVocabulary.load(args.load_vocab)
        loop_detector = LoopDetector(
            vocab=vocab, camera_K=camera.K,
            on_loop_detected=on_loop, min_bow_score=0.012,
            min_geo_inliers=25, consistency=3, temporal_window=20,
            vocab_build_at=50, verbose=True,
        )
        if saved_map and saved_map.recognizer:
            loop_detector._recognizer = saved_map.recognizer
        loop_detector.start()

    def on_new_keyframe(kf):
        if mapper and not args.reloc_only: mapper.enqueue(kf)
        if loop_detector: loop_detector.enqueue(kf)
    vo.on_new_keyframe = on_new_keyframe

    # ── Source ────────────────────────────────────────────────────────────
    if args.kitti:      source = kitti_loader(args.kitti)
    elif args.video:    source = video_loader(args.video)
    elif args.webcam:   source = webcam_loader()
    else:
        print("No source — running synthetic scene demo.")
        from demo import SyntheticScene
        source = SyntheticScene(camera)

    show_gui     = not args.no_gui
    blurry_count = 0
    MAP_LOG_EVERY = 5

    print(f"\nRunning. Keys: q=quit  s=snapshot  l=loops  r=reloc")
    if _HAS_RERUN:
        print("Rerun viewer: RMB-drag=orbit  scroll=zoom  WASD=fly\n")

    # ── Main loop ─────────────────────────────────────────────────────────
    for img_l, img_r, fid in source:

        if is_blurry(img_l, threshold=args.blur_threshold):
            blurry_count += 1
            if show_gui: cv2.waitKey(1)
            continue

        # Relocalization
        if not reloc_done and relocalizer is not None:
            gray_l       = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            orb_r        = cv2.ORB_create(nfeatures=1000)
            kps_l, descs_l = orb_r.detectAndCompute(gray_l, None)
            if descs_l is not None and len(descs_l) > 20:
                pts_l  = np.array([kp.pt for kp in kps_l], dtype=np.float32)
                result = relocalizer.relocalize(descs_l, pts_l)
                if result.success:
                    vo.T_world_cam  = result.T_world_cam.copy()
                    reloc_status    = f"OK  KF{result.matched_kf_id}"
                    reloc_successes += 1
                    reloc_done      = True
            reloc_attempts += 1
            if reloc_attempts >= args.reloc_frames and not reloc_done:
                reloc_status = f"failed ({reloc_attempts} attempts)"
                reloc_done   = True

        stats = vo.process(img_l, img_right=img_r, timestamp=fid / 30.0)

        # Per-frame Rerun log
        rr_logger.log_frame(
            fid=fid, frame_bgr=img_l, T_world_cam=vo.T_world_cam,
            cur_feats=vo._last_features, stats=stats, vo_state=vo.state.name,
        )

        # Map update (throttled)
        if fid % MAP_LOG_EVERY == 0:
            traj = vo.trajectory if len(vo.trajectory) > 1 else np.zeros((1, 3))
            rr_logger.log_map(vo.map_points, vo.keyframes, traj, loop_events)

        # Minimal OpenCV status window
        if show_gui:
            status = np.zeros((50, 700, 3), dtype=np.uint8)
            txt = (f"F{fid:4d}  matched={stats.num_matched:4d}  "
                   f"inliers={stats.num_inliers:4d}  MPs={len(vo.map_points):5d}  "
                   f"{stats.process_ms:.0f}ms  {'[KF]' if stats.is_keyframe else ''}")
            cv2.putText(status, txt, (8, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("VSLAM  (3-D scene in Rerun)", status)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if fid % 10 == 0:
            pos = vo.T_world_cam[:3, 3]
            print(f"  Frame {fid:4d} | matched={stats.num_matched:4d} | "
                  f"inliers={stats.num_inliers:4d} | MPs={len(vo.map_points):5d} | "
                  f"pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) | "
                  f"{stats.process_ms:.1f}ms"
                  + (" [KF]" if stats.is_keyframe else ""))

    # ── Shutdown ──────────────────────────────────────────────────────────
    if mapper:       mapper.stop()
    if loop_detector: loop_detector.stop()

    if args.save_vocab and loop_detector and loop_detector._recognizer:
        loop_detector._recognizer.vocab.save(args.save_vocab)
    if args.save_map and not args.reloc_only:
        MapStorage.save(path=args.save_map, vo=vo, loop_detector=loop_detector)

    out_path_poses = "results_demo_rerun.txt"
    with open(out_path_poses, "w") as f:
        for T in vo.pose_graph.poses:
            f.write(" ".join(f"{v:.6e}" for v in T[:3].flatten()) + "\n")

    # Final full-map push to Rerun
    if _HAS_RERUN and len(vo.trajectory) > 1:
        rr_logger.log_map(vo.map_points, vo.keyframes, vo.trajectory, loop_events)
        print("\nRerun: final map pushed.  Keep the viewer open to explore.")

    print("\n" + vo.summary())
    print(f"=== Run Summary ===")
    print(f"  Blurry frames : {blurry_count}")
    print(f"  Loop closures : {len(loop_events)}")
    print(f"  PGO runs      : {pgo.n_optimizations}")
    print(f"  Poses saved   : {out_path_poses}")
    if mapper:
        print(f"  BA runs       : {mapper.n_ba_runs}")
        print(f"  Points culled : {mapper.n_pts_culled}")

    if show_gui:
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════ #
#  CLI                                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VSLAM Demo with Rerun 3-D scene reconstruction"
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--kitti",  type=str)
    src.add_argument("--video",  type=str)
    src.add_argument("--webcam", action="store_true")

    parser.add_argument("--fx", type=float); parser.add_argument("--fy", type=float)
    parser.add_argument("--cx", type=float); parser.add_argument("--cy", type=float)
    parser.add_argument("--hfov", type=float)

    parser.add_argument("--no-local-map",    action="store_true")
    parser.add_argument("--no-loop",         action="store_true")
    parser.add_argument("--save-vocab",      type=str)
    parser.add_argument("--load-vocab",      type=str)
    parser.add_argument("--save-map",        type=str)
    parser.add_argument("--load-map",        type=str)
    parser.add_argument("--reloc-frames",    type=int, default=30)
    parser.add_argument("--reloc-only",      action="store_true")
    parser.add_argument("--min-loop-gap",    type=int, default=200)
    parser.add_argument("--output",          type=str, default="trajectory.png")
    parser.add_argument("--no-gui",          action="store_true")
    parser.add_argument("--blur-threshold",  type=float, default=80.0)

    rr_group = parser.add_mutually_exclusive_group()
    rr_group.add_argument("--rr-spawn",   action="store_true", default=True,
                           help="Spawn a local Rerun viewer (default)")
    rr_group.add_argument("--rr-connect", action="store_true",
                           help="Connect to a running viewer on localhost:9876")
    rr_group.add_argument("--rr-save",    type=str, metavar="FILE",
                           help="Write a .rrd file for offline replay")

    args = parser.parse_args()
    run(args)