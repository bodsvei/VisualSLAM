# VisualSLAM Project — Gemini Handover Document

**Last updated:** June 2026 (Refined for Stereo + Robustness)  
**Repo:** `~/Documents/GitHub/VisualSLAM/`  
**Python:** 3.9, macOS (Apple Silicon — Bikrams-Laptop)  
**Package name:** `vo_slam`

---

## 1. Project Overview

A Python-based Visual SLAM system evolving from monocular to stereo capability. Target: StellaVSLAM-level performance on KITTI and real-world OAK-D Lite data.

**Major Milestone Reached:** The system now supports **Stereo Vision** (image_0/image_1) with direct metric baseline triangulation.

---

## 2. Coordinate Frame & Pose Accumulation

**CRITICAL: Mathematically sound pose chain (FIXED)**
```
Camera frame  : OpenCV standard (+X right, +Y down, +Z forward)
World frame   : = camera frame at t=0 (Identity at frame 0)

T_{ref <- cur}: Relative motion from reference to current camera.
                Obtained via cv2.recoverPose(ref, cur).

Accumulation  : T_{world <- cur} = T_{world <- ref} @ T_{ref <- cur}
                (NO inversion needed if using correct multiplication order)
```

---

## 3. Module Status

### `camera.py` — CameraModel (Updated)
Now supports `baseline` and `bf` (baseline * fx). Methods: `project()`, `project_right()`, `undistort_points()`, `backproject()`. Factories: `kitti()` (seq 00), `from_matrix()`.

### `features.py` — FeatureDetector + Matcher (Updated)
- **CLAHE**: Integrated Contrast Limited Adaptive Histogram Equalization for light robustness.
- **Stereo Matcher**: `match_stereo()` matches left to right using epipolar (y-axis) and positive disparity constraints.

### `triangulation.py` — Triangulator (Updated)
- **Disparity Mapping**: `triangulate_stereo()` computes metric 3D points from a single frame using $Z = bf / disparity$.
- **Observations**: `MapPoint.obs` dict `{kf_id: feature_index}` is the source of truth for BA/PGO.

### `pipeline.py` — VisualOdometry (Restored & Robust)
Orchestrates tracking and mapping. All "fucked up" regressions from unanchored BA have been reverted.
- **Thread Safety**: `self.lock` (threading.Lock) protects all shared state (poses, MPs, KFs).
- **Metric Logic**: Supports stereo initialization and frame-by-frame metric recovery.
- **Sync**: `process()` ensures 1:1 frame-to-pose mapping in `pose_graph`.

### `local_mapping.py` — LocalMapper (Robust)
- **BA Propogation**: Corrections from Local BA are now propagated to the **entire trajectory spine** (non-keyframes) in `vo.pose_graph`.
- **Tracker Sync**: Updates `vo.T_world_cam` after BA to prevent "jumping back" kinks.

### `bundle_adjustment.py` — Local BA (Reverted to Stable)
- **Strategy C (Scipy)**: Reverted to the stable version. Fixed-keyframe anchoring and Huber kernels were removed as they caused instability in STAGE-10.

### `pose_graph_optimizer.py` — PGO (Optimized)
- **Global Snap**: When a loop is closed, `_update_map_and_graph` rigid-transforms **all map points** and **all poses** in the spine.
- **Safety**: Uses `PoseGraph.transform_all()` for atomic, locked batch updates.

---

## 4. Operational Guide

### Running KITTI (Stereo)
```bash
python3 run_kitti.py --kitti KITTI --seq 00 --load-vocab vocab_00.pkl
```
*Note: Ensure `image_0` and `image_1` exist in the sequence folder.*

### Evaluation Results (Expected)
- **Scale Factor**: Should be **~1.0** (Stereo metric fix).
- **Trajectory**: Should be smooth (Spine propagation fix) and correctly angled (Pose chain fix).

### Debugging & Tools
- `diagnose.py`: Post-run JSON analyzer.
- `stereo_calibration.py`: Precise checkerboard calibration for OAK-D Lite.

---

## 5. Key Architecture Invariants

1. **Thread Isolation**: The tracker (frontend) MUST NOT be blocked by the optimizer (backend). Use `vo.lock` for all shared memory access.
2. **Spine Consistency**: Always update `vo.pose_graph` when keyframes move. The visual "jaggedness" was solved by drawing all poses, not just sparse keyframes.
3. **Metric Scale**: In stereo mode, trust disparity over monocular triangulation.

---

## 6. Revert History
- **Regressions in STAGE-10**: Caused by "anchoring" attempt in Scipy BA and unverified "Huber" parameters. Reverted to early stable Stereo version.
- **Coordinate Inversions**: `invert_pose()` was removed and then re-added depending on the specific `recoverPose` usage. The current `pipeline.py` uses the re-added stable inversions.