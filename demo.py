"""
demo.py
-------
Run the Visual Odometry pipeline.

Modes
-----
  1. KITTI dataset  – pass --kitti /path/to/sequence/image_0
                      optionally --calib and --gt-poses for full evaluation
  2. Video file     – pass --video  path/to/video.mp4
  3. Webcam         – pass --webcam

Usage examples
--------------
  python demo.py                                       # synthetic
  python demo.py --kitti /data/kitti/00/image_0        # KITTI frames only
  python demo.py --kitti /data/kitti/00/image_0 \
                 --calib    /data/kitti/00/calib.txt \
                 --gt-poses /data/kitti/poses/00.txt   # full eval + evo cmds
  python demo.py --video  input.mp4
  python demo.py --webcam

Stage 2 / Stage 3 flags (require vo_slam.local_mapping / vo_slam.loop_detector)
  --no-loop              Disable loop detection (Stage 3)
  --save-vocab FILE      Save BoW vocabulary to .pkl after run
  --load-vocab FILE      Load pre-built BoW vocabulary from .pkl

Press 'q' to quit, 's' to save a trajectory snapshot.
"""

import argparse
import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import matplotlib

if os.environ.get("DISPLAY") or sys.platform == "darwin" or sys.platform == "win32":
    try:
        matplotlib.use("TkAgg")   # interactive — supports plt.show()
    except Exception:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")         # headless server — no display
import matplotlib.pyplot as plt

# ── Add parent directory so `vo_slam` is importable when running directly ── #
sys.path.insert(0, str(Path(__file__).parent))

from vo_slam import (
    CameraModel, VOConfig, VisualOdometry, DetectorType,
    FeatureOverlay, TrajectoryPlot, plot_trajectory_static,
)

# ── Stage 2 / 3 optional imports ─────────────────────────────────────────── #
try:
    from vo_slam.local_mapping import LocalMapper
    from vo_slam.loop_detector import LoopDetector, LoopEvent
    from vo_slam.vocabulary    import VisualVocabulary
    _HAS_STAGES_23 = True
except ImportError:
    _HAS_STAGES_23 = False


# ═══════════════════════════════════════════════════════════════════════════ #
#  KITTI helpers                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

def load_calib(path: Path) -> np.ndarray:
    """Parse calib.txt; return the 3×3 intrinsic matrix K extracted from P0."""
    with open(path) as f:
        for line in f:
            if line.startswith("P0:"):
                vals = list(map(float, line.strip().split()[1:]))
                return np.array(vals).reshape(3, 4)[:3, :3]
    raise ValueError("P0 not found in calib.txt")


def load_poses(path: Path):
    """Load KITTI-format ground-truth poses; return a list of 4×4 matrices."""
    poses = []
    with open(path) as f:
        for line in f:
            T = np.array(list(map(float, line.strip().split()))).reshape(3, 4)
            poses.append(np.vstack([T, [0, 0, 0, 1]]))
    return poses

# ═══════════════════════════════════════════════════════════════════════════ #
#  Dataset loaders                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def kitti_loader(folder: str):
    """Yield (image, frame_id) from a KITTI image folder."""
    p = Path(folder)
    files = sorted(p.glob("*.png")) + sorted(p.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No images in {folder}")
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is not None:
            yield img, i


def video_loader(path: str):
    """Yield (frame, frame_id) from a video file."""
    cap = cv2.VideoCapture(path)
    fid = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame, fid
        fid += 1
    cap.release()


def webcam_loader(device: int = 0):
    """Yield (frame, frame_id) from webcam."""
    cap = cv2.VideoCapture(device)
    fid = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame, fid
        fid += 1
    cap.release()


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main runner                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

def run(args):
    # ── Timestamp ────────────────────────────────────────────────────────── #
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_tag = ("kitti"     if args.kitti  else
                  "video"     if args.video  else
                  "webcam"    if args.webcam else
                  "synthetic")

    # JSON output path (KITTI mode only)
    json_path = None
    if args.kitti:
        json_dir  = Path("logs")
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / f"run_{source_tag}_{timestamp}.json"

    print(f"{'='*60}")
    print(f"  Visual SLAM Demo — source: {source_tag}  [{timestamp}]")
    if json_path:
        print(f"  JSON : {json_path}")
    print(f"{'='*60}")

    # ── Camera ───────────────────────────────────────────────────────────── #
    if args.kitti:
        if args.calib:
            K      = load_calib(Path(args.calib))
            camera = CameraModel.from_matrix(K, width=1241, height=376)
            print(f"Camera: loaded from {args.calib}")
            print(f"K matrix:\n{camera.K}")
        else:
            camera = CameraModel.kitti()
            print("Camera: using KITTI preset (pass --calib for exact K)")

    elif args.webcam or args.video:
        cap = cv2.VideoCapture(0 if args.webcam else args.video)
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if args.fx and args.fy and args.cx and args.cy:
            K = np.array([
                [args.fx, 0,       args.cx],
                [0,       args.fy, args.cy],
                [0,       0,       1.0    ],
            ])
            camera = CameraModel.from_matrix(K, width=W, height=H)
            print(f"Camera: provided intrinsics  fx={args.fx}  fy={args.fy}  "
                  f"cx={args.cx}  cy={args.cy}")

        elif args.hfov:
            camera = CameraModel.from_fov(args.hfov, W, H)
            print(f"Camera: hfov={args.hfov}° → fx≈{camera.fx:.1f}")

        else:
            camera = CameraModel.from_fov(60.0, W, H)
            print("[WARNING] No camera intrinsics provided; using hfov=60° estimate.")
            print("          For better results pass --fx --fy --cx --cy or --hfov.")

    else:
        camera = CameraModel.from_fov(70.0, width=1241, height=376)

    # ── Ground-truth poses (optional, KITTI only) ─────────────────────────── #
    gt = None
    if args.gt_poses:
        gt = load_poses(Path(args.gt_poses))
        print(f"GT poses: {len(gt)} loaded from {args.gt_poses}")

    # ── VO config ────────────────────────────────────────────────────────── #
    cfg = VOConfig(
        detector_type   = DetectorType.ORB,
        max_features    = 1500,
        ratio_thresh    = 0.75,
        min_inliers     = 15,
        scale_mode      = "fixed",
        store_images    = False,
        kf_min_parallax = 2.0,
        kf_min_frames   = 3,
        kf_max_frames   = 15,
    )
    vo = VisualOdometry(camera, cfg)
    print(f"VOConfig: scale_mode={cfg.scale_mode}  max_features={cfg.max_features}  "
          f"min_inliers={cfg.min_inliers}")

    # ── Stage 2: Local Mapper ─────────────────────────────────────────────── #
    mapper               = None
    loop_detector        = None
    loop_events_received = []

    if _HAS_STAGES_23:
        mapper = LocalMapper(
            camera         = camera,
            map_points_ref = vo.map_points,
            keyframes_ref  = vo.keyframes,
            verbose        = False,
        )

        # ── Stage 3: Loop Detector ───────────────────────────────────────── #
        if not args.no_loop:
            def on_loop(event: LoopEvent):
                loop_events_received.append({
                    "query_kf_id": event.query_kf_id,
                    "match_kf_id": event.match_kf_id,
                    "bow_score"  : round(float(event.bow_score), 4),
                    "geo_inliers": int(event.geo_inliers),
                    "timestamp"  : datetime.now().isoformat(),
                })
                print(
                    f"*** LOOP CLOSURE ***  "
                    f"KF{event.query_kf_id} <-> KF{event.match_kf_id}  "
                    f"geo_inliers={event.geo_inliers}  bow={event.bow_score:.4f}"
                )

            vocab = None
            if args.load_vocab and Path(args.load_vocab).exists():
                vocab = VisualVocabulary.load(args.load_vocab)
                print(f"Vocabulary loaded from {args.load_vocab}")

            loop_detector = LoopDetector(
                vocab            = vocab,
                camera_K         = camera.K,
                on_loop_detected = on_loop,
                min_bow_score    = 0.012,
                min_geo_inliers  = 25,
                consistency      = 3,
                temporal_window  = 20,
                vocab_build_at   = 50,
                verbose          = True,
            )

        # ── Wire on_new_keyframe hook ────────────────────────────────────── #
        def on_new_keyframe(kf):
            mapper.enqueue(kf)
            if loop_detector is not None:
                loop_detector.enqueue(kf)

        vo.on_new_keyframe = on_new_keyframe

        # ── Start background threads ─────────────────────────────────────── #
        mapper.start()
        if loop_detector is not None:
            loop_detector.start()

        print(
            f"Stage 2 (LocalMapper): active  |  "
            f"Stage 3 (LoopDetector): "
            f"{'active' if loop_detector is not None else 'disabled (--no-loop)'}"
        )
    else:
        print("[INFO] vo_slam.local_mapping / loop_detector not found; Stage 1 only.")

    # ── Source ───────────────────────────────────────────────────────────── #
    if args.kitti:
        source = kitti_loader(args.kitti)
    elif args.video:
        source = video_loader(args.video)
    elif args.webcam:
        source = webcam_loader()
    else:
        print("No source specified – running synthetic scene demo.")
        source = SyntheticScene(camera)

    # ── Visualisation setup ───────────────────────────────────────────────── #
    traj_plotter = TrajectoryPlot(figsize=(500, 400))
    show_gui     = not args.no_gui

    # ── Standalone ORB tracker (mirrors orb.py) ───────────────────────────── #
    orb_detector = cv2.ORB_create(nfeatures=1000)
    orb_matcher  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    orb_prev_frame: np.ndarray | None = None
    orb_prev_kp                        = None
    orb_prev_des                       = None

    # ── Per-frame accumulators ────────────────────────────────────────────── #
    frame_log   = []      # one entry per logged frame (every 50 when GT is loaded)
    all_poses   = []      # one 4×4 per processed frame, used for pose file output
    last_good_T = np.eye(4)
    t_run_start = time.perf_counter()

    # ── Table header — printed only when GT is loaded ─────────────────────── #
    if gt is not None:
        print()
        print('─' * 90)
        print(
            f"{'Frame':>6}  {'est_x':>8} {'est_z':>8}  "
            f"{'gt_x':>8} {'gt_z':>8}  "
            f"{'KFs':>5} {'MPs':>7} {'Loops':>6} {'State':<8} {'ms':>6}"
        )
        print('─' * 90)

    print("\nRunning VO. Press 'q' to quit, 's' to save trajectory.\n")

    # ── Main processing loop ──────────────────────────────────────────────── #
    for frame, fid in source:
        t0    = time.perf_counter()
        stats = vo.process(frame, timestamp=fid / 30.0)
        dt_ms = (time.perf_counter() - t0) * 1000

        # Track best-known pose (guard against NaN on tracking-lost frames)
        if np.isfinite(vo.T_world_cam).all():
            last_good_T = vo.T_world_cam.copy()
        all_poses.append(last_good_T.copy())

        if show_gui:
            # Camera feed with HUD
            display = FeatureOverlay.draw_hud(
                frame,
                frame_id   = fid,
                num_kf     = len(vo.keyframes),
                num_mp     = len(vo.map_points),
                position   = vo.T_world_cam[:3, 3],
                is_kf      = stats.is_keyframe,
                process_ms = stats.process_ms,
            )
            cv2.imshow("Camera", display)

            # Trajectory map (rendered every 5 frames to save time)
            if fid % 5 == 0 and len(vo.trajectory) > 1:
                traj_img = traj_plotter.render_2d(
                    vo.trajectory, vo.map_points, vo.keyframes
                )
                cv2.imshow("Trajectory", traj_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                _save_snapshot(vo, fid)

        # ── Console logging ──────────────────────────────────────────────── #
        if gt is not None:
            # Table-style every 50 frames with side-by-side GT comparison
            if fid % 50 == 0:
                pos    = last_good_T[:3, 3]
                gt_idx = min(fid, len(gt) - 1)
                gt_pos = gt[gt_idx][:3, 3]

                print(
                    f"{fid:>6}  "
                    f"{pos[0]:>8.2f} {pos[2]:>8.2f}  "
                    f"{gt_pos[0]:>8.2f} {gt_pos[2]:>8.2f}  "
                    f"{len(vo.keyframes):>5} {len(vo.map_points):>7} "
                    f"{len(loop_events_received):>6} {vo.state.name:<8} {dt_ms:>6.1f}"
                )
                frame_log.append({
                    "frame"      : fid,
                    "est_x"      : round(float(pos[0]), 3),
                    "est_y"      : round(float(pos[1]), 3),
                    "est_z"      : round(float(pos[2]), 3),
                    "gt_x"       : round(float(gt_pos[0]), 3),
                    "gt_z"       : round(float(gt_pos[2]), 3),
                    "keyframes"  : len(vo.keyframes),
                    "map_points" : len(vo.map_points),
                    "loops"      : len(loop_events_received),
                    "state"      : vo.state.name,
                    "process_ms" : round(dt_ms, 2),
                    "inliers"    : stats.num_inliers,
                    "matched"    : stats.num_matched,
                })
        else:
            # Original every-10-frames style (non-GT modes)
            if fid % 10 == 0:
                pos    = vo.T_world_cam[:3, 3]
                kf_tag = " [KF]" if stats.is_keyframe else ""
                print(
                    f"  Frame {fid:4d} | "
                    f"matched={stats.num_matched:4d} | "
                    f"inliers={stats.num_inliers:4d} | "
                    f"MPs={len(vo.map_points):5d} | "
                    f"pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) | "
                    f"{stats.process_ms:.1f}ms{kf_tag}"
                )

    total_time = time.perf_counter() - t_run_start

    # ── Stop background threads ───────────────────────────────────────────── #
    if mapper is not None:
        mapper.stop()
    if loop_detector is not None:
        loop_detector.stop()

    # ── Save vocabulary ───────────────────────────────────────────────────── #
    if (args.save_vocab
            and loop_detector is not None
            and hasattr(loop_detector, "_recognizer")
            and loop_detector._recognizer):
        loop_detector._recognizer.vocab.save(args.save_vocab)
        print(f"Vocabulary saved → {args.save_vocab}")

    # ── Save KITTI-format pose file ───────────────────────────────────────── #
    out_poses = None
    if args.kitti and all_poses:
        out_poses = f"results_{timestamp}.txt"
        with open(out_poses, "w") as f:
            for T in all_poses:
                f.write(" ".join(f"{v:.6e}" for v in T[:3].flatten()) + "\n")

    # ── Final summary ─────────────────────────────────────────────────────── #
    print('─' * 60)
    print()
    print("=== Run Summary ===")
    print(f"  Source            : {source_tag}")
    print(f"  Total frames      : {len(all_poses)}")
    print(f"  Keyframes         : {len(vo.keyframes)}")
    print(f"  Map points        : {len(vo.map_points)}")
    print(f"  Loop closures     : {len(loop_events_received)}")
    print(f"  Final state       : {vo.state.name}")
    print(f"  Total runtime     : {total_time:.1f}s  "
          f"({len(all_poses) / total_time:.1f} fps avg)")
    if out_poses:
        print(f"  Poses saved       : {out_poses}  ({len(all_poses)} lines)")
    if json_path:
        print(f"  JSON saved        : {json_path}")

    # ── evo evaluation commands (printed when GT is available) ────────────── #
    if gt is not None and out_poses:
        print()
        print("=== Evaluation Commands ===")
        print(f"  evo_ape kitti {args.gt_poses} {out_poses} \\")
        print(f"      --plot_mode xz --save_plot logs/ate_{timestamp}.pdf \\")
        print(f"      --save_results logs/ate_{timestamp}.zip")
        print()
        print(f"  evo_traj kitti {args.gt_poses} {out_poses} \\")
        print(f"      --ref {args.gt_poses} --plot_mode xz \\")
        print(f"      --save_plot logs/traj_{timestamp}.pdf")
        print()
        print(f"  evo_rpe kitti {args.gt_poses} {out_poses} \\")
        print(f"      --delta 100 --delta_unit m \\")
        print(f"      --save_plot logs/rpe_{timestamp}.pdf")

    # ── Save structured JSON (KITTI mode only) ────────────────────────────── #
    if json_path is not None:
        json_data = {
            "meta": {
                "source"          : source_tag,
                "timestamp"       : timestamp,
                "total_frames"    : len(all_poses),
                "total_runtime_s" : round(total_time, 2),
                "avg_fps"         : round(len(all_poses) / total_time, 2),
            },
            "config": {
                "scale_mode"    : cfg.scale_mode,
                "max_features"  : cfg.max_features,
                "min_inliers"   : cfg.min_inliers,
                "kf_min_frames" : cfg.kf_min_frames,
                "kf_max_frames" : cfg.kf_max_frames,
            },
            "camera": {
                "fx": round(float(camera.fx), 4),
                "fy": round(float(camera.fy), 4),
                "cx": round(float(camera.cx), 4),
                "cy": round(float(camera.cy), 4),
            },
            "results": {
                "keyframes"      : len(vo.keyframes),
                "map_points"     : len(vo.map_points),
                "loop_closures"  : len(loop_events_received),
                "final_state"    : vo.state.name,
                "poses_file"     : out_poses,
            },
            "loop_events" : loop_events_received,
            "frame_log"   : frame_log,
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

    print("Done.")

    # ── Final trajectory plot ─────────────────────────────────────────────── #
    if len(vo.trajectory) > 1:
        out_path = args.output or "trajectory.png"
        fig = plot_trajectory_static(
            vo.trajectory, vo.map_points, vo.keyframes,
            save_path=out_path,
            title=f"VO Trajectory ({len(vo.trajectory)} frames, "
                  f"{len(vo.keyframes)} keyframes)",
        )
        print(f"\nTrajectory saved to: {out_path}")
        if show_gui and matplotlib.get_backend() != "Agg":
            plt.show()
        else:
            print(f"Trajectory saved to: {out_path}")
        plt.close(fig)

    if show_gui:
        cv2.destroyAllWindows()


def _save_snapshot(vo, fid):
    path = f"snapshot_frame{fid:04d}.png"
    if len(vo.trajectory) > 1:
        fig = plot_trajectory_static(
            vo.trajectory, vo.map_points, vo.keyframes,
            save_path=path,
            title=f"Snapshot @ frame {fid}",
        )
        plt.close(fig)
        print(f"  → Snapshot saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════ #
#  CLI                                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Odometry Demo")

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--kitti",  type=str, help="Path to KITTI image folder")
    src.add_argument("--video",  type=str, help="Path to video file")
    src.add_argument("--webcam", action="store_true", help="Use webcam (device 0)")

    # KITTI evaluation extras
    parser.add_argument("--calib",    type=str, default=None,
                        help="Path to KITTI calib.txt (enables exact K instead of preset)")
    parser.add_argument("--gt-poses", type=str, default=None, dest="gt_poses",
                        help="Path to KITTI GT poses file; enables table output and evo cmds")

    # Output
    parser.add_argument("--output", type=str, default="trajectory.png",
                        help="Output trajectory image path")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable OpenCV windows (headless mode)")

    # Camera intrinsics (video / webcam)
    parser.add_argument("--fx",   type=float, help="Focal length x (pixels)")
    parser.add_argument("--fy",   type=float, help="Focal length y (pixels)")
    parser.add_argument("--cx",   type=float, help="Principal point x (pixels)")
    parser.add_argument("--cy",   type=float, help="Principal point y (pixels)")
    parser.add_argument("--hfov", type=float,
                        help="Horizontal FOV in degrees (fallback if fx/fy not given)")

    # Stage 2 / 3 flags (ported from run_kitti.py)
    parser.add_argument("--no-loop",    action="store_true",
                        help="Disable loop detection (Stage 3)")
    parser.add_argument("--save-vocab", type=str, default=None, dest="save_vocab",
                        help="Save BoW vocabulary to .pkl after run")
    parser.add_argument("--load-vocab", type=str, default=None, dest="load_vocab",
                        help="Load pre-built BoW vocabulary from .pkl")

    args = parser.parse_args()
    run(args)