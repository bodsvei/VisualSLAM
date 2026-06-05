"""
run_kitti.py
------------
KITTI evaluation script with Stage 2 (Local Mapping) + Stage 3 (Place Recognition).
All output is written to both stdout and a timestamped log file.

Usage
-----
  python3 run_kitti.py                          # seq 00
  python3 run_kitti.py --seq 05                # different sequence
  python3 run_kitti.py --no-loop               # disable loop detection
  python3 run_kitti.py --save-vocab vocab.pkl  # save vocabulary for reuse
  python3 run_kitti.py --load-vocab vocab.pkl  # load pre-built vocabulary

Log files
---------
  logs/run_<seq>_<timestamp>.log   — full console output
  logs/run_<seq>_<timestamp>.json  — structured stats (ATE-ready)
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

sys.path.insert(0, str(Path(__file__).parent))

from vo_slam                import CameraModel, VisualOdometry, VOConfig, DetectorType
from vo_slam.local_mapping  import LocalMapper
from vo_slam.loop_detector  import LoopDetector, LoopEvent
from vo_slam.vocabulary     import VisualVocabulary


# ═══════════════════════════════════════════════════════════════════════ #
#  Logger setup — writes to stdout AND log file simultaneously            #
# ═══════════════════════════════════════════════════════════════════════ #

def setup_logger(log_path: Path) -> logging.Logger:
    """
    Returns a logger that mirrors all output to both:
      - terminal (stdout)
      - log_path (file)
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("KITTI_VO")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt     = "[%(asctime)s] %(message)s",
        datefmt = "%H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

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

    # ── Log file paths ───────────────────────────────────────────────── #
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
    K      = load_calib(CALIB_FILE)
    gt     = load_poses(GT_FILE)
    camera = CameraModel.from_matrix(K, width=1241, height=376)
    log.info(f"Camera : {camera}")
    log.info(f"K matrix:\n{camera.K}")

    # ── VO config ────────────────────────────────────────────────────── #
    cfg = VOConfig(
        detector_type = DetectorType.ORB,
        max_features  = 2000,
        min_inliers   = 20,
        scale_mode    = "fixed",
        fixed_scale   = 1.0,
        kf_min_frames = 3,
        kf_max_frames = 20,
    )
    vo = VisualOdometry(camera, cfg)
    log.info(f"VOConfig: scale_mode={cfg.scale_mode}  max_features={cfg.max_features}  "
             f"min_inliers={cfg.min_inliers}")

    # ── Stage 2: Local Mapper ─────────────────────────────────────────── #
    mapper = LocalMapper(
        camera         = camera,
        map_points_ref = vo.map_points,
        keyframes_ref  = vo.keyframes,
        verbose        = False,
    )

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
            f"*** LOOP CLOSURE ***  "
            f"KF{event.query_kf_id} <-> KF{event.match_kf_id}  "
            f"geo_inliers={event.geo_inliers}  bow={event.bow_score:.4f}"
        )

    vocab = None
    if args.load_vocab and Path(args.load_vocab).exists():
        vocab = VisualVocabulary.load(args.load_vocab)
        log.info(f"Vocabulary loaded from {args.load_vocab}")

    loop_detector = LoopDetector(
        vocab            = vocab,
        camera_K         = camera.K,
        on_loop_detected = on_loop if not args.no_loop else None,
        min_bow_score    = 0.012,
        min_geo_inliers  = 25,
        consistency      = 3,
        temporal_window  = 20,
        vocab_build_at   = 50,
        verbose          = True,
    )

    # ── Wire hooks ────────────────────────────────────────────────────── #
    def on_new_keyframe(kf):
        mapper.enqueue(kf)
        if not args.no_loop:
            loop_detector.enqueue(kf)

    vo.on_new_keyframe = on_new_keyframe

    # ── Start threads ─────────────────────────────────────────────────── #
    mapper.start()
    if not args.no_loop:
        loop_detector.start()

    # ── Per-frame stats accumulator ───────────────────────────────────── #
    frame_log   = []   # one entry per logged frame (every 50)
    t_run_start = time.perf_counter()

    # ── Run ───────────────────────────────────────────────────────────── #
    images      = sorted(IMG_DIR.glob("*.png"))
    last_good_T = np.eye(4)
    all_poses   = []

    log.info(f"")
    log.info(f"Sequence {SEQ}: {len(images)} frames  |  {len(gt)} GT poses")
    log.info(f"{'─'*90}")
    log.info(
        f"{'Frame':>6}  {'est_x':>8} {'est_z':>8}  "
        f"{'gt_x':>8} {'gt_z':>8}  "
        f"{'KFs':>5} {'MPs':>7} {'Loops':>6} {'State':<8} {'ms':>6}"
    )
    log.info(f"{'─'*90}")

    for i, img_path in enumerate(images):
        img   = cv2.imread(str(img_path))
        t0    = time.perf_counter()
        stats = vo.process(img, timestamp=i / 10.0)
        dt_ms = (time.perf_counter() - t0) * 1000

        if np.isfinite(vo.T_world_cam).all():
            last_good_T = vo.T_world_cam.copy()
        all_poses.append(last_good_T.copy())

        if i % 50 == 0:
            pos    = last_good_T[:3, 3]
            gt_pos = gt[i][:3, 3]

            row = (
                f"{i:>6}  "
                f"{pos[0]:>8.2f} {pos[2]:>8.2f}  "
                f"{gt_pos[0]:>8.2f} {gt_pos[2]:>8.2f}  "
                f"{len(vo.keyframes):>5} {len(vo.map_points):>7} "
                f"{len(loop_events_received):>6} {vo.state.name:<8} {dt_ms:>6.1f}"
            )
            log.info(row)

            frame_log.append({
                "frame"      : i,
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

    total_time = time.perf_counter() - t_run_start

    # ── Stop threads ──────────────────────────────────────────────────── #
    mapper.stop()
    if not args.no_loop:
        loop_detector.stop()

    # ── Save vocabulary ───────────────────────────────────────────────── #
    if args.save_vocab and loop_detector._recognizer:
        loop_detector._recognizer.vocab.save(args.save_vocab)
        log.info(f"Vocabulary saved → {args.save_vocab}")

    # ── Save poses ────────────────────────────────────────────────────── #
    out_path = f"results_{SEQ}.txt"
    with open(out_path, "w") as f:
        for T in all_poses:
            f.write(" ".join(f"{v:.6e}" for v in T[:3].flatten()) + "\n")

    assert len(all_poses) == len(images), \
        f"Pose count mismatch: {len(all_poses)} vs {len(images)}"

    # ── Final summary ─────────────────────────────────────────────────── #
    log.info(f"{'─'*90}")
    log.info(f"")
    log.info(f"=== Run Summary ===")
    log.info(f"  Sequence          : {SEQ}")
    log.info(f"  Total frames      : {len(images)}")
    log.info(f"  Keyframes         : {len(vo.keyframes)}")
    log.info(f"  Map points        : {len(vo.map_points)}")
    log.info(f"  Loop closures     : {len(loop_events_received)}")
    log.info(f"  Final state       : {vo.state.name}")
    log.info(f"  Total runtime     : {total_time:.1f}s  ({len(images)/total_time:.1f} fps avg)")
    log.info(f"  Poses saved       : {out_path}  ({len(all_poses)} lines)")
    log.info(f"  Log saved         : {log_path}")
    log.info(f"  JSON saved        : {json_path}")
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

    # ── Save structured JSON ──────────────────────────────────────────── #
    json_data = {
        "meta": {
            "sequence"      : SEQ,
            "timestamp"     : timestamp,
            "total_frames"  : len(images),
            "total_runtime_s": round(total_time, 2),
            "avg_fps"       : round(len(images) / total_time, 2),
        },
        "config": {
            "scale_mode"    : cfg.scale_mode,
            "fixed_scale"   : cfg.fixed_scale,
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
            "keyframes"         : len(vo.keyframes),
            "map_points"        : len(vo.map_points),
            "loop_closures"     : len(loop_events_received),
            "final_state"       : vo.state.name,
            "poses_file"        : out_path,
        },
        "loop_events"   : loop_events_received,
        "frame_log"     : frame_log,
    }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    log.info(f"Done.")


# ═══════════════════════════════════════════════════════════════════════ #
#  CLI                                                                    #
# ═══════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="KITTI VO evaluation — Stage 2+3")
    p.add_argument("--kitti",      default="KITTI",    help="KITTI root directory")
    p.add_argument("--seq",        default="00",       help="Sequence ID (00–10)")
    p.add_argument("--no-loop",    action="store_true",help="Disable loop detection")
    p.add_argument("--save-vocab", type=str,           help="Save vocabulary to .pkl")
    p.add_argument("--load-vocab", type=str,           help="Load vocabulary from .pkl")
    run(p.parse_args())