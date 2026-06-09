"""
diagnose.py
-----------
Run this after a KITTI sequence to surface all four bugs from Analysis 01+02.
Prints a clear report of what's working and what's broken.

Usage
-----
  python3 diagnose.py --log logs/run_00_<timestamp>.json
  python3 diagnose.py --live   # run seq 00 and diagnose in real time
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ═══════════════════════════════════════════════════════════════════════ #
#  Analyze from saved JSON log                                            #
# ═══════════════════════════════════════════════════════════════════════ #

def diagnose_from_json(json_path: str):
    with open(json_path) as f:
        data = json.load(f)

    print("\n" + "═"*60)
    print("  VSLAM DIAGNOSTIC REPORT")
    print("═"*60)

    meta    = data.get("meta", {})
    results = data.get("results", {})
    config  = data.get("config", {})
    camera  = data.get("camera", {})
    frames  = data.get("frame_log", [])
    loops   = data.get("loop_events", [])

    print(f"\n── Run Info ──────────────────────────────────────────────")
    print(f"  Sequence     : {meta.get('sequence', '?')}")
    print(f"  Total frames : {meta.get('total_frames', '?')}")
    print(f"  Runtime      : {meta.get('total_runtime_s', '?')}s  "
          f"({meta.get('avg_fps', '?')} fps)")

    # ── Check 1: Initialization height ──────────────────────────────── #
    print(f"\n── Check 1: Initialization Height ────────────────────────")
    if frames:
        first = frames[0]
        y0 = first.get("est_y", 0.0)
        print(f"  First logged pose Y : {y0:.3f}m")
        if abs(y0) > 5.0:
            print(f"  ✗ FAIL — Y starts at {y0:.1f}m instead of ~0m")
            print(f"    → Check pipeline.py _initialize(): T_world_cam must be np.eye(4)")
            print(f"    → Check triangulation initial baseline is not tilted")
        else:
            print(f"  ✓ OK — Y ≈ 0 at initialization")
    else:
        print("  (no frame log data)")

    # ── Check 2: Loop closure events ────────────────────────────────── #
    print(f"\n── Check 2: Loop Closure Events ──────────────────────────")
    n_loops = len(loops)
    print(f"  Loop events fired    : {n_loops}")

    if n_loops == 0:
        print(f"  ✗ FAIL — No loops detected at all")
        print(f"    → Local drift may be too severe for BoW to match")
        print(f"    → Check loop_detector.summary() at end of run")
        print(f"    → Try: lower min_bow_score (0.012→0.008), lower min_geo_inliers (30→20)")
    elif n_loops > 50:
        print(f"  ✗ WARN — Loop flooding: {n_loops} events (expected 1–5 for seq 00)")
        print(f"    → Fix: set min_loop_gap_frames=200 in LoopDetector")
        print(f"    → Fix: enable match-KF deduplication (_used_match_kf_ids)")

        # Check for temporal clustering
        if len(loops) > 2:
            frames_with_loops = [ev.get("timestamp", "") for ev in loops]
            print(f"    → First loop : {loops[0]}")
            print(f"    → Last loop  : {loops[-1]}")
    else:
        print(f"  ✓ OK — {n_loops} loops detected (healthy for seq 00)")
        for ev in loops[:3]:
            print(f"    KF{ev['query_kf_id']} ↔ KF{ev['match_kf_id']}  "
                  f"bow={ev['bow_score']}  inliers={ev['geo_inliers']}")

    # ── Check 3: Local BA running ────────────────────────────────────── #
    print(f"\n── Check 3: Local Bundle Adjustment ──────────────────────")
    # Detect from frame_log whether MPs grew (sign BA is triangulating+culling)
    if len(frames) > 3:
        mp_start = frames[0].get("map_points", 0)
        mp_mid   = frames[len(frames)//2].get("map_points", 0)
        mp_end   = frames[-1].get("map_points", 0)
        print(f"  Map points: start={mp_start}  mid={mp_mid}  end={mp_end}")

        if mp_end < 100:
            print(f"  ✗ FAIL — Very few map points ({mp_end}). "
                  f"BA is likely not running.")
            print(f"    → Check g2o: python3 -c 'import g2o; print(g2o.__version__)'")
            print(f"    → If not installed: pip install g2o-python")
            print(f"    → Check BA anchor: oldest KF in window must be fixed")
        elif mp_start > mp_end and mp_end < 500:
            print(f"  ✗ WARN — Map points shrinking aggressively (culling too strict?)")
        else:
            print(f"  ✓ OK — Map points growing normally")
    else:
        print("  (insufficient frame log data)")

    # ── Check 4: Drift profile ───────────────────────────────────────── #
    print(f"\n── Check 4: Trajectory Drift Profile ─────────────────────")
    if len(frames) > 4:
        # Compute rough per-segment drift as distance from GT
        drifts = []
        for fr in frames:
            ex = fr.get("est_x", 0)
            ez = fr.get("est_z", 0)
            gx = fr.get("gt_x", 0)
            gz = fr.get("gt_z", 0)
            drifts.append(np.sqrt((ex-gx)**2 + (ez-gz)**2))

        print(f"  Drift at  0%  of seq : {drifts[0]:.1f}m")
        print(f"  Drift at 25%  of seq : {drifts[len(drifts)//4]:.1f}m")
        print(f"  Drift at 50%  of seq : {drifts[len(drifts)//2]:.1f}m")
        print(f"  Drift at 100% of seq : {drifts[-1]:.1f}m")

        # Pattern analysis
        if drifts[-1] > drifts[len(drifts)//2] * 1.5:
            print(f"  ✗ Drift accelerating — BA not correcting local errors")
        elif all(d < 20 for d in drifts):
            print(f"  ✓ Drift appears bounded (<20m) — healthy")

    # ── Check 5: g2o availability ────────────────────────────────────── #
    print(f"\n── Check 5: Backend Dependencies ─────────────────────────")
    try:
        import g2o
        print(f"  ✓ g2o available: {g2o.__version__}")
    except ImportError:
        print(f"  ✗ g2o NOT installed — BA and PGO are running in no-op mode")
        print(f"    → Install: pip install g2o-python")

    try:
        from scipy.optimize import minimize
        print(f"  ✓ scipy available — PGO fallback ready")
    except ImportError:
        print(f"  ✗ scipy NOT installed")
        print(f"    → Install: pip install scipy")

    print(f"\n── Recommended Next Steps ────────────────────────────────")
    print(f"  1. pip install g2o-python scipy")
    print(f"  2. Replace loop_detector.py with fixed version (dead zone)")
    print(f"  3. Replace bundle_adjustment.py with fixed version (anchor KF)")
    print(f"  4. Add PoseGraphOptimizer to run_kitti.py")
    print(f"  5. Re-run and check that BA runs > 0 in the summary")
    print("═"*60 + "\n")


# ═══════════════════════════════════════════════════════════════════════ #
#  Live diagnostic (runs a short segment and checks state)               #
# ═══════════════════════════════════════════════════════════════════════ #

def diagnose_live(args):
    import cv2
    from vo_slam import CameraModel, VisualOdometry, VOConfig, DetectorType
    from vo_slam.local_mapping import LocalMapper
    from vo_slam.loop_detector import LoopDetector

    SEQ        = args.seq
    KITTI_ROOT = Path(args.kitti)
    IMG_DIR    = KITTI_ROOT / "sequences" / SEQ / "image_0"
    CALIB_FILE = KITTI_ROOT / "sequences" / SEQ / "calib.txt"

    def load_calib(path):
        with open(path) as f:
            for line in f:
                if line.startswith("P0:"):
                    vals = list(map(float, line.strip().split()[1:]))
                    return np.array(vals).reshape(3, 4)[:3, :3]

    K      = load_calib(CALIB_FILE)
    camera = CameraModel.from_matrix(K, width=1241, height=376)
    cfg    = VOConfig(detector_type=DetectorType.ORB, max_features=2000,
                      min_inliers=20, scale_mode="fixed")
    vo     = VisualOdometry(camera, cfg)

    mapper = LocalMapper(camera=camera, map_points_ref=vo.map_points,
                         keyframes_ref=vo.keyframes, verbose=True)
    loop_det = LoopDetector(vocab=None, camera_K=camera.K,
                             min_geo_inliers=30, vocab_build_at=50)

    vo.on_new_keyframe = lambda kf: (mapper.enqueue(kf), loop_det.enqueue(kf))
    mapper.start()
    loop_det.start()

    images = sorted(IMG_DIR.glob("*.png"))[:args.n_frames]
    print(f"Running {len(images)} frames for live diagnosis...\n")

    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        vo.process(img)

    mapper.stop()
    loop_det.stop()

    print("\n" + "═"*60)
    print("  LIVE DIAGNOSTIC")
    print("═"*60)
    print(f"\nVO Summary:")
    print(vo.summary())
    print(f"\nLocal Mapper:")
    print(f"  BA runs      : {mapper.n_ba_runs}")
    print(f"  Pts culled   : {mapper.n_pts_culled}")
    print(f"  KFs culled   : {mapper.n_kfs_culled}")
    print(f"\nLoop Detector:")
    print(f"  {loop_det.summary()}")

    # Check initialization
    if len(vo.keyframes) > 0:
        first_pos = vo.keyframes[0].position
        print(f"\nFirst KF position: {first_pos.round(4)}")
        if np.linalg.norm(first_pos) > 0.1:
            print("  ✗ First KF is NOT at origin — check initialization")
        else:
            print("  ✓ First KF is at origin")

    # Check BA
    if mapper.n_ba_runs == 0:
        print("\n✗ BA ran ZERO times — check g2o install and KF/MP thresholds")
    else:
        print(f"\n✓ BA ran {mapper.n_ba_runs} times")

    print("═"*60 + "\n")


# ═══════════════════════════════════════════════════════════════════════ #
#  CLI                                                                    #
# ═══════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VSLAM Diagnostic Tool")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--log",  type=str, help="Path to run JSON log file")
    mode.add_argument("--live", action="store_true", help="Run live diagnosis")
    p.add_argument("--kitti",    default="KITTI",  help="KITTI root (for --live)")
    p.add_argument("--seq",      default="00",     help="Sequence (for --live)")
    p.add_argument("--n-frames", default=500, type=int,
                   help="Frames to process in live mode")
    args = p.parse_args()

    if args.log:
        diagnose_from_json(args.log)
    else:
        diagnose_live(args)