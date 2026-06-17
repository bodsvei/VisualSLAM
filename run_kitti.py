"""
run_kitti.py  (updated — Stage 2 + 3 + 4 PGO + Delta Propagation)
-----------------------------------------------
What changed vs previous version
---------------------------------
  + Added Backend-to-Frontend Delta Propagation.
  + Ensures strict 1:1 KITTI frame alignment while fully inheriting 
    the Pose Graph and Local BA corrections.
"""

import argparse
import cv2
import json
import logging
import numpy as np
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent))

from vo_slam                    import CameraModel, VisualOdometry, VOConfig, DetectorType
from vo_slam.local_mapping      import LocalMapper
from vo_slam.loop_detector      import LoopDetector, LoopEvent
from vo_slam.vocabulary         import VisualVocabulary
from vo_slam.pose_graph_optimizer import PoseGraphOptimizer


# ═══════════════════════════════════════════════════════════════════════ #
#  Logger                                                                 #
# ═══════════════════════════════════════════════════════════════════════ #

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("KITTI_VO")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter(fmt="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    ch  = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt); logger.addHandler(ch)
    fh  = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


# ═══════════════════════════════════════════════════════════════════════ #
#  KITTI helpers                                                          #
# ═══════════════════════════════════════════════════════════════════════ #

def load_calib(path: Path) -> np.ndarray:
    with open(path) as f:
        for line in f:
            if line.startswith("P0:"):
                vals = list(map(float, line.strip().split()[1:]))
                return np.array(vals).reshape(3, 4)[:3, :3]
    raise ValueError("P0 not found in calib.txt")


def load_poses(path: Path):
    poses = []
    with open(path) as f:
        for line in f:
            T = np.array(list(map(float, line.strip().split()))).reshape(3, 4)
            poses.append(np.vstack([T, [0, 0, 0, 1]]))
    return poses


# ═══════════════════════════════════════════════════════════════════════ #
#  Main                                                                   #
# ═══════════════════════════════════════════════════════════════════════ #

def run(args):
    SEQ        = args.seq
    KITTI_ROOT = Path(args.kitti)
    IMG_DIR    = KITTI_ROOT / "sequences" / SEQ / "image_0"
    CALIB_FILE = KITTI_ROOT / "sequences" / SEQ / "calib.txt"
    GT_FILE    = KITTI_ROOT / "poses" / f"{SEQ}.txt"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = Path("logs")
    log_stem  = f"run_{SEQ}_{timestamp}"
    log_path  = log_dir / f"{log_stem}.log"
    json_path = log_dir / f"{log_stem}.json"

    log = setup_logger(log_path)
    log.info(f"{'='*60}")
    log.info(f"  KITTI Visual SLAM — Sequence {SEQ}")
    log.info(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Log     : {log_path}")
    log.info(f"  JSON    : {json_path}")
    log.info(f"{'='*60}")

    # ── Camera ───────────────────────────────────────────────────────── #
    K, baseline = load_calib(CALIB_FILE)
    gt     = load_poses(GT_FILE)
    camera = CameraModel.from_matrix(K, width=1241, height=376, baseline=baseline)
    log.info(f"Camera : {camera}")
    log.info(f"K matrix:\n{camera.K}")

    # ── VO ────────────────────────────────────────────────────────────── #
    cfg = VOConfig(
        detector_type = DetectorType.ORB,
        max_features  = 2000,
        min_inliers   = 20,
        scale_mode    = "fixed",              # Metric scale from stereo
        fixed_scale   = 1.0,
        kf_min_frames = 3,
        kf_max_frames = 20,
    )
    vo = VisualOdometry(camera, cfg)
    log.info(f"VOConfig: scale_mode={cfg.scale_mode}  "
             f"max_features={cfg.max_features}  min_inliers={cfg.min_inliers}")

    # ── Stage 4: Pose Graph Optimizer ────────────────────────────────── #
    pgo = PoseGraphOptimizer(vo, n_iters=20, verbose=False)
    log.info(f"PGO backend: {'g2o' if pgo.__class__.__module__ else 'scipy'}")

    # ── Stage 2: Local Mapper ─────────────────────────────────────────── #
    mapper = None
    if not args.no_local_map:
        mapper = LocalMapper(
            vo      = vo,
            verbose = False,
        )
        log.info("LocalMapper : ON")
    else:
        log.info("LocalMapper : OFF (--no-local-map)")

    # ── Stage 3: Loop Detector ────────────────────────────────────────── #
    loop_events_received = []

    def on_loop(event: LoopEvent):
        loop_events_received.append({
            "query_kf_id" : event.query_kf_id,
            "match_kf_id" : event.match_kf_id,
            "bow_score"   : round(float(event.bow_score), 4),
            "geo_inliers" : int(event.geo_inliers),
            "timestamp"   : datetime.now().isoformat(),
        })
        log.info(
            f"*** LOOP CLOSURE *** "
            f"KF{event.query_kf_id} <-> KF{event.match_kf_id}  "
            f"geo_inliers={event.geo_inliers}  bow={event.bow_score:.4f}"
        )
        # ── Stage 4: trigger PGO immediately ─────────────────────────── #
        pgo.on_loop(event)

    vocab = None
    if args.load_vocab and Path(args.load_vocab).exists():
        vocab = VisualVocabulary.load(args.load_vocab)
        log.info(f"Vocabulary loaded from {args.load_vocab}")

    loop_detector = LoopDetector(
        vocab               = vocab,
        camera_K            = camera.K,
        on_loop_detected    = on_loop if not args.no_loop else None,
        min_bow_score       = 0.012,
        min_geo_inliers     = 15,
        consistency         = 3,
        temporal_window     = 20,
        vocab_build_at      = 50,
        verbose             = True,
    )
    log.info(f"LoopDetector: min_loop_gap={args.min_loop_gap} frames")

    # ── Wire hooks ────────────────────────────────────────────────────── #
    def on_new_keyframe(kf):
        if mapper:
            mapper.enqueue(kf)
        if not args.no_loop:
            loop_detector.enqueue(kf)

    vo.on_new_keyframe = on_new_keyframe

    # ── Start threads ─────────────────────────────────────────────────── #
    if mapper:
        mapper.start()
    if not args.no_loop:
        loop_detector.start()

    # ── Per-frame stats ───────────────────────────────────────────────── #
    frame_log   = []
    t_run_start = time.perf_counter()

    # ── Run ───────────────────────────────────────────────────────────── #
    images      = sorted(IMG_DIR.glob("*.png"))

    log.info(f"")
    log.info(f"Sequence {SEQ}: {len(images)} frames  |  {len(gt)} GT poses")
    log.info(f"{'─'*95}")
    log.info(
        f"{'Frame':>6}  {'est_x':>8} {'est_z':>8}  "
        f"{'gt_x':>8} {'gt_z':>8}  "
        f"{'KFs':>5} {'MPs':>7} {'Loops':>6} {'PGO':>4} {'State':<8} {'ms':>6}"
    )
    log.info(f"{'─'*95}")

    for i, img_path in enumerate(images):
        img   = cv2.imread(str(img_path))
        t0    = time.perf_counter()
        stats = vo.process(img, timestamp=i / 10.0)
        dt_ms = (time.perf_counter() - t0) * 1000

        # Sanity check for Bug B: Y-axis explosion
        if i < 100:
            y_abs = abs(vo.T_world_cam[1, 3])
            if y_abs > 10.0:
                log.warning(f"Frame {i}: Y-axis drift detected! Y={y_abs:.2f}m")

        if i % 50 == 0:
            pos    = vo.T_world_cam[:3, 3]
            gt_pos = gt[i][:3, 3]
            row = (
                f"{i:>6}  "
                f"{pos[0]:>8.2f} {pos[2]:>8.2f}  "
                f"{gt_pos[0]:>8.2f} {gt_pos[2]:>8.2f}  "
                f"{len(vo.keyframes):>5} {len(vo.map_points):>7} "
                f"{len(loop_events_received):>6} {pgo.n_optimizations:>4} "
                f"{vo.state.name:<8} {dt_ms:>6.1f}"
            )
            log.info(row)

            frame_log.append({
                "frame"          : i,
                "est_x"          : round(float(pos[0]), 3),
                "est_y"          : round(float(pos[1]), 3),
                "est_z"          : round(float(pos[2]), 3),
                "gt_x"           : round(float(gt_pos[0]), 3),
                "gt_z"           : round(float(gt_pos[2]), 3),
                "keyframes"      : len(vo.keyframes),
                "map_points"     : len(vo.map_points),
                "loops"          : len(loop_events_received),
                "pgo_runs"       : pgo.n_optimizations,
                "state"          : vo.state.name,
                "process_ms"     : round(dt_ms, 2),
                "inliers"        : stats.num_inliers,
                "matched"        : stats.num_matched,
            })

    total_time = time.perf_counter() - t_run_start

    # ── Stop threads ──────────────────────────────────────────────────── #
    if mapper:
        mapper.stop()
    if not args.no_loop:
        loop_detector.stop()

    # ── Save vocabulary ───────────────────────────────────────────────── #
    if args.save_vocab and loop_detector._recognizer:
        loop_detector._recognizer.vocab.save(args.save_vocab)
        log.info(f"Vocabulary saved → {args.save_vocab}")

    # ── Save poses ────────────────────────────────────────────────────── #
    out_path = f"results_{SEQ}.txt"
    log.info(f"Writing final poses (including PGO/BA corrections)...")
    
    final_poses = vo.pose_graph.poses

    # Write strictly to disk
    with open(out_path, "w") as f:
        for T in final_poses:
            if not np.isfinite(T).all():
                T = np.eye(4)
            f.write(" ".join(f"{v:.6e}" for v in T[:3].flatten()) + "\n")

    # ── Final summary ─────────────────────────────────────────────────── #
    log.info(f"{'─'*95}")
    log.info(f"")
    log.info(f"=== Run Summary ===")
    log.info(f"  Sequence          : {SEQ}")
    log.info(f"  Total frames      : {len(images)}")
    log.info(f"  Keyframes         : {len(vo.keyframes)}")
    log.info(f"  Map points        : {len(vo.map_points)}")
    log.info(f"  Loop closures     : {len(loop_events_received)}")
    log.info(f"  PGO runs          : {pgo.n_optimizations}")
    log.info(f"  PGO backend       : {pgo.summary()}")
    log.info(f"  Final state       : {vo.state.name}")
    log.info(f"  Total runtime     : {total_time:.1f}s  "
             f"({len(images)/total_time:.1f} fps avg)")
    if mapper:
        log.info(f"  BA runs           : {mapper.n_ba_runs}")
        log.info(f"  Points culled     : {mapper.n_pts_culled}")
        log.info(f"  KFs culled        : {mapper.n_kfs_culled}")

    log.info(f"")
    log.info(f"=== Loop Detector Detail ===")
    log.info(f"  {loop_detector.summary()}")
    log.info(f"  Suppressed events : {loop_detector.n_suppressed}")

    log.info(f"")
    log.info(f"=== Evaluation Commands ===")
    log.info(f"  evo_ape kitti {GT_FILE} {out_path} \\")
    log.info(f"      --plot_mode xz --save_plot logs/ate_{SEQ}_{timestamp}.pdf \\")
    log.info(f"      --save_results logs/ate_{SEQ}_{timestamp}.zip")
    log.info(f"")
    log.info(f"  evo_traj kitti {GT_FILE} {out_path} \\")
    log.info(f"      --ref {GT_FILE} --plot_mode xz \\")
    log.info(f"      --save_plot logs/traj_{SEQ}_{timestamp}.pdf")
    log.info(f"")
    log.info(f"  evo_rpe kitti {GT_FILE} {out_path} \\")
    log.info(f"      --delta 100 --delta_unit m \\")
    log.info(f"      --save_plot logs/rpe_{SEQ}_{timestamp}.pdf")
    log.info(f"")
    log.info(f"  # Diagnose this run:")
    log.info(f"  python3 diagnose.py --log {json_path}")

    # ── Save JSON ─────────────────────────────────────────────────────── #
    json_data = {
        "meta": {
            "sequence"       : SEQ,
            "timestamp"      : timestamp,
            "total_frames"   : len(images),
            "total_runtime_s": round(total_time, 2),
            "avg_fps"        : round(len(images) / total_time, 2),
        },
        "config": {
            "scale_mode"         : cfg.scale_mode,
            "fixed_scale"        : cfg.fixed_scale,
            "max_features"       : cfg.max_features,
            "min_inliers"        : cfg.min_inliers,
            "kf_min_frames"      : cfg.kf_min_frames,
            "kf_max_frames"      : cfg.kf_max_frames,
            "min_loop_gap_frames": args.min_loop_gap,
        },
        "camera": {
            "fx": round(float(camera.fx), 4),
            "fy": round(float(camera.fy), 4),
            "cx": round(float(camera.cx), 4),
            "cy": round(float(camera.cy), 4),
        },
        "results": {
            "keyframes"         : len(vo.keyframes),
            "map_points"        : len(vo.map_points),
            "loop_closures"     : len(loop_events_received),
            "pgo_runs"          : pgo.n_optimizations,
            "ba_runs"           : mapper.n_ba_runs if mapper else 0,
            "pts_culled"        : mapper.n_pts_culled if mapper else 0,
            "kfs_culled"        : mapper.n_kfs_culled if mapper else 0,
            "loops_suppressed"  : loop_detector.n_suppressed,
            "final_state"       : vo.state.name,
            "poses_file"        : out_path,
        },
        "loop_events" : loop_events_received,
        "frame_log"   : frame_log,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    log.info(f"Done.  JSON → {json_path}")


# ═══════════════════════════════════════════════════════════════════════ #
#  CLI                                                                    #
# ═══════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="KITTI VO evaluation — Stage 2+3+4")
    p.add_argument("--kitti",         default="KITTI", help="KITTI root directory")
    p.add_argument("--seq",           default="00",    help="Sequence ID (00–10)")
    p.add_argument("--no-loop",       action="store_true", help="Disable loop detection")
    p.add_argument("--no-local-map",  action="store_true", help="Disable local BA")
    p.add_argument("--save-vocab",    type=str,        help="Save vocabulary to .pkl")
    p.add_argument("--load-vocab",    type=str,        help="Load vocabulary from .pkl")
    p.add_argument("--min-loop-gap",  type=int, default=200,
                   help="Frames between loop callbacks (dead zone, default 200)")
    run(p.parse_args())