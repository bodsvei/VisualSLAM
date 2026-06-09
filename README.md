# Visual Odometry & SLAM — Python

A modular monocular VO/V-SLAM pipeline designed as a foundation for a full SLAM system.  
Long-term target: **StellaVSLAM-level** capability, reached incrementally.

**Current Status:** Stage 2 (Local Mapping) — Core VO functional with bug fixes applied.

---

## 🚨 Security Vulnerabilities & Recommendations

### CRITICAL: Unsafe Deserialization (pickle.load)

**Location:** `vo_slam/map_storage.py:170`, `vo_slam/vocabulary.py:194`

**Risk:** Both files use `pickle.load()` to deserialize data from disk without any validation. Pickle is inherently unsafe and can execute arbitrary code during deserialization. An attacker who can write to the map file or vocabulary file could achieve remote code execution.

**Exploit Scenario:**
```python
# Malicious pickle payload that executes code on load
import pickle, os
class Exploit:
    def __reduce__(self):
        return (os.system, ("rm -rf /important/data",))
pickle.dump(Exploit(), open("malicious_map.pkl", "wb"))
```

**Fix Required:**
1. **Short-term:** Add file integrity checks (SHA-256 hash verification) before loading
2. **Medium-term:** Replace pickle with a safe serialization format:
   - Use JSON + NumPy arrays saved separately (`.npz`)
   - Use MessagePack for binary data
   - Use protocol buffers or flatbuffers for structured data
3. **Long-term:** Implement signed map files with cryptographic verification

**Recommended Change for `map_storage.py`:**
```python
import hashlib
import json

# Instead of pickle:
def save(path, vo, loop_detector):
    data = {
        "camera_K": vo.camera.K.tolist(),
        "keyframes": [...],  # Convert to JSON-serializable format
        "metadata": {...}
    }
    with open(path + ".json", "w") as f:
        json.dump(data, f)
    np.savez(path + ".npz", **numpy_arrays)
    # Save SHA-256 checksum
    compute_and_save_hash(path)
```

### HIGH: eval() Usage in Debug Script

**Location:** `check_g2o.py:51`

**Risk:** The script uses `eval(code)` to probe g2o solvers. While currently only used locally for debugging, this pattern should never be in production code.

**Fix:** Replace with direct attribute access:
```python
# Instead of eval(code):
solver_class = getattr(g2o, solver_name)
solver_instance = solver_class()
```

### MEDIUM: Unvalidated File Paths

**Location:** Multiple files accept user-provided paths without sanitization

**Risk:** Path traversal attacks if user input is not properly validated.

**Affected:**
- `demo.py --load-map`, `--load-vocab`
- `run_kitti.py KITTI_ROOT`
- `map_storage.py load/save`

**Fix:** Use `pathlib.Path.resolve()` and validate against allowed directories.

---

## Architecture

```
VisualSLAM/
├── vo_slam/               ← main Python package (renamed from 'vo' — see Platform Notes)
│   ├── __init__.py        ← exports all public classes
│   ├── camera.py          ← CameraModel  (intrinsics, projection, back-projection)
│   ├── features.py        ← FeatureDetector + FeatureMatcher (ORB/SIFT, BF/FLANN, LK flow)
│   ├── motion.py          ← MotionEstimator (Essential matrix → R, t via RANSAC)
│   ├── triangulation.py   ← Triangulator (DLT + reprojection filter → MapPoints)
│   ├── keyframe.py        ← Keyframe dataclass + KeyframeSelector policy
│   ├── pipeline.py        ← VisualOdometry orchestrator + PoseGraph
│   └── visualization.py   ← FeatureOverlay, TrajectoryPlot, VOVisualizer
├── demo.py                ← Runnable demo (synthetic / KITTI / video / webcam)
├── run_kitti.py           ← KITTI evaluation script (outputs results_00.txt)
└── tests/test_vo.py       ← 25 unit tests (pytest), all passing
```

---

## ⚠️ Known Bugs

**Read this before running anything.**

### BUG 1 — Coordinate frame inversion `[FIXED ✓]`

**Status:** Fixed in `pipeline.py` (lines 405-416)

`cv2.recoverPose()` returns `T_{ref←cur}` but the original code composed it as `T_{cur←ref}`. The fix inverts R,t before composing:
```python
R_ref_cur = pose.R.T
t_ref_cur = -(pose.R.T @ (pose.t.ravel() * scale))
```

### BUG 2 — Scale explosion with `median_depth` mode `[FIXED ✓]`

**Status:** Fixed in `pipeline.py` (lines 99-100, 419-426, 529-540)

`_recover_scale()` now clamps depth to [0.3, 80.0] metres and includes a pose health check that reverts to the last good pose if the result is non-finite or exceeds `pose_max_norm` (1e5).

### BUG 3 — Map points frozen at ~46 `[FIXED ✓ - downstream of BUG 2]`

**Status:** Resolved once BUG 2 fix is applied. Triangulation baseline validity check added (lines 432-453).

### BUG 4 — `evo --plot` blank on macOS `[WORKAROUND]`

Use `--save_plot trajectory.pdf` instead of `--plot`. Or install `pyqt5` and set `MPLBACKEND=Qt5Agg`.

---

## Methodology

1. **Feature Detection & Matching** — ORB features by default with grid-based suppression (4×4 cells) to ensure uniform keypoint distribution. Matched with Brute-Force Hamming distance, filtered by Lowe's ratio test (threshold 0.75) and optional symmetric cross-check.

2. **Motion Estimation** — Relative pose estimated via the Essential Matrix (`cv2.findEssentialMat`, RANSAC, 5-point algorithm). Decomposed into R and t using `cv2.recoverPose`, which includes a chirality check so triangulated points lie in front of both cameras. A Homography score is computed in parallel to flag planar/degenerate scenes (H_score > 0.45 triggers a warning).

3. **Scale Recovery** — Monocular VO has no metric scale. `scale_mode='fixed'` (default, recommended) keeps unit-norm translation. `scale_mode='median_depth'` normalises translation so newly triangulated points maintain a consistent median depth — experimentally useful but currently unstable (BUG 2). Do not use `median_depth` until BUG 1 and BUG 2 are fixed.

4. **Pose Accumulation** — Poses use the **camera-to-world** convention (`T_world_cam`, 4×4 SE3). Absolute pose is composed by right-multiplication: `T_world_cam_new = T_world_cam_prev × T_{cur←ref}`. Note: `recoverPose` returns `T_{ref←cur}` — it must be inverted before composing (BUG 1 is a violation of this).

5. **Keyframe Selection** — Pose is estimated against the last **keyframe**, not the immediately previous frame, giving a larger baseline and better triangulation. New keyframes are triggered by parallax, feature survival ratio, rotation magnitude, or a forced interval cap.

---

## Quick Start

```bash
# Install editably (prevents namespace collision with system 'vo' package)
pip install -e .
pip install opencv-python numpy matplotlib pytest evo pyqt5

# Verify
python -m pytest tests/ -v         # should show 25 passed

# Synthetic flythrough (no data needed)
python demo.py

# Video file
python demo.py --video my_drive.mp4

# Webcam
python demo.py --webcam

# Headless (no GUI, save trajectory to file)
python demo.py --no-gui --output traj.png
```

---

## Testing on KITTI

### Step 1 — Download the dataset

Register (free) at **http://www.cvlibs.net/datasets/kitti/eval_odometry.php** and download:
- `data_odometry_gray.zip` — grayscale image sequences (22 GB)
- `data_odometry_poses.zip` — ground truth poses
- `data_odometry_calib.zip` — calibration files

> **Tip:** Browser downloads use a single TCP connection with no parallelism. `aria2c` splits into chunks, downloads them simultaneously, and uses all available bandwidth:
> ```bash
> aria2c -x 16 -s 16 <url>
> ```

Extract so the layout is:

```
KITTI/
├── sequences/
│   └── 00/
│       ├── image_0/     ← left grayscale camera (4541 frames)
│       ├── calib.txt
│       └── times.txt
└── poses/
    └── 00.txt           ← ground truth SE3 poses (3×4 row-major)
```

### Step 2 — Run the evaluation script

Update `KITTI_ROOT` in `run_kitti.py` to point to your dataset, then:

```bash
python3 run_kitti.py     # outputs results_00.txt (must have exactly 4541 lines)
```

### Step 3 — Evaluate with `evo`

```bash
# ATE — Absolute Trajectory Error
evo_ape kitti KITTI/poses/00.txt results_00.txt \
    --plot_mode xz --save_plot ate.pdf --save_results ate.zip

# RPE — Relative Pose Error per 100m segment
evo_rpe kitti KITTI/poses/00.txt results_00.txt \
    --delta 100 --delta_unit m --save_plot rpe.pdf

# Trajectory overlay vs ground truth
evo_traj kitti KITTI/poses/00.txt results_00.txt \
    --ref KITTI/poses/00.txt --plot_mode xz --save_plot traj.pdf
```

> **macOS:** Use `--save_plot file.pdf`, not `--plot`. The interactive matplotlib window is broken on macOS without a Qt backend. See BUG 4.

### Expected ATE benchmarks (Sequence 00)

| Stage | Expected ATE |
|-------|-------------|
| BUG 1+2 fixed, `scale_mode='fixed'` | ~100–300 m |
| + Local BA (Stage 2) | ~20–50 m |
| + Loop closure (Stage 4) | ~5–15 m |
| StellaVSLAM full system | ~5 m |

---

## Roadmap

```
Stage 1: Solid VO          ← COMPLETE (BUG 1-3 fixed)
      │
      ▼
Stage 2: Local Mapping     ← IN PROGRESS
Stage 3: Place Recognition ← BoW vocabulary implemented
Stage 4: Loop Closing
Stage 5: Map Reuse         ← Map save/load implemented (security fix needed)
Stage 6: Full SLAM System
```

### Stage 1 — Visual Odometry front-end *(COMPLETE ✓)*
ORB tracking, Essential matrix, triangulation, and pose graph are all in place and tested. BUG 1 (coordinate inversion), BUG 2 (scale explosion), and BUG 3 (triangulation with bad pose) have been fixed.

### Stage 2 — Local Mapping & Bundle Adjustment *(IN PROGRESS)*
Prevents short-term drift from accumulating.
- **CovisibilityGraph** — nodes = keyframes, edges weighted by shared map point count. Wire to `vo.on_new_keyframe` callback.
- **Local Bundle Adjustment** — jointly optimise poses and map points over a local keyframe window using `g2o-python`.
- **Map Culling** — prune weak map points (high reprojection error, few observations) and redundant keyframes.

### Stage 3 — Place Recognition *(PARTIAL)*
Detect loop candidates using a **Bag of Words** vocabulary tree. Vocabulary and BoW database modules are implemented (`vocabulary.py`, `bow_database.py`, `place_recognition.py`).

### Stage 4 — Loop Closing
- **Sim3 Solver** — compute a 7-DOF transform (R, t, scale) to correct accumulated scale drift at a detected loop.
- **Pose-graph Optimisation** — propagate the Sim3 correction through the essential graph to update all poses efficiently.

### Stage 5 — Map Reuse & Relocalization *(PARTIAL)*
- **Relocalization** — if tracking goes LOST, match against the BoW database and recover pose via PnP.
- **Map Save/Load** — serialise map, poses, and BoW database to disk for multi-session mapping. **⚠️ SECURITY WARNING:** Current implementation uses unsafe `pickle.load()` — see Security section above.

### Stage 6 — Full System Integration
Three-thread architecture (Tracking / Local Mapping / Loop Closing) + ROS topic bridge for fusion with the RGBD perception stack.

---

## Programmatic Usage

```python
from vo_slam import CameraModel, VisualOdometry, VOConfig, DetectorType

camera = CameraModel.kitti()   # KITTI seq 00 intrinsics

cfg = VOConfig(
    detector_type = DetectorType.ORB,
    max_features  = 1500,
    ratio_thresh  = 0.75,
    scale_mode    = "fixed",   # recommended for stability
)

vo = VisualOdometry(camera, cfg)

for frame in my_frame_source:
    stats = vo.process(frame)
    T    = vo.T_world_cam    # 4×4 SE3 pose (camera → world)
    traj = vo.trajectory     # (N, 3) accumulated positions

# VSLAM back-end hook — fires on every new keyframe
vo.on_new_keyframe = lambda kf: my_loop_closer.submit(kf)
```

---

## Configuration Reference

All parameters live in `VOConfig` (`pipeline.py`).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detector_type` | `ORB` | `ORB` / `SIFT` / `FAST_ORB` |
| `max_features` | `2000` | Max keypoints retained per frame |
| `grid_rows` | `4` | Grid suppression rows |
| `grid_cols` | `4` | Grid suppression columns |
| `matcher_type` | `BF_HAMMING` | `BF_HAMMING` / `BF_L2` / `FLANN` |
| `ratio_thresh` | `0.75` | Lowe's ratio test threshold |
| `ransac_thresh` | `1.0` | RANSAC reprojection threshold (px) |
| `ransac_prob` | `0.999` | RANSAC confidence |
| `min_inliers` | `20` | Minimum RANSAC inliers to accept a pose |
| `scale_mode` | `'fixed'` | `'fixed'` (safe) / `'median_depth'` (experimental) / `'none'` |
| `fixed_scale` | `1.0` | Translation norm when `scale_mode='fixed'` |
| `max_reproj_err` | `2.0` | Triangulation reprojection filter (px) |
| `min_parallax_deg` | `1.0` | Minimum parallax angle for triangulation |
| `min_depth` | `0.1` | Minimum valid map point depth (m) |
| `max_depth` | `200.0` | Maximum valid map point depth (m) |
| `kf_min_parallax` | `2.0` | Min pixel displacement to trigger a new keyframe |
| `kf_max_feat_ratio` | `0.75` | Feature survival ratio below which a KF is forced |
| `kf_max_rot_deg` | `15.0` | Rotation threshold to force a new keyframe |
| `kf_min_frames` | `3` | Minimum frames between keyframes |
| `kf_max_frames` | `20` | Maximum frames before a keyframe is forced |

---

## Coordinate Convention

```
Camera frame : OpenCV standard — +X right, +Y down, +Z forward (into scene)
World frame  : = camera frame at t=0 (identity at first frame)

T_world_cam  : 4×4 SE3 — transforms points from camera → world
               p_world = T_world_cam @ p_cam

recoverPose  : returns T_{ref←cur}   ←  INVERSE of what you want
               Must invert before composing into T_world_cam

Accumulation : T_world_cam_new = T_world_cam_old × T_{cur←ref}
```

> **Important:** `cv2.recoverPose` output convention is a known source of bugs. This has been fixed in the current codebase — see BUG 1 fix above.

---

## Platform Notes (macOS, Apple Silicon)

- **Package name must be `vo_slam`**, not `vo`. A system-level namespace package named `vo` exists at an unknown path and silently shadows a local `vo/` directory. Always use `pip install -e .` (editable install) to ensure the correct package is on `sys.path`.
- **Python 3.9+** is recommended.
- **matplotlib interactive windows** do not work reliably — use `--save_plot file.pdf` with all `evo` commands, and `--no-gui` or `--output file.png` with `demo.py` if the display hangs.
- **g2o-python:** For bundle adjustment, install `g2o-python`. Run `python check_g2o.py` to probe available solver types on your system.

---

## Additional Improvement Suggestions

### Code Quality

1. **Type Hints:** Add comprehensive type hints throughout the codebase for better IDE support and static analysis.

2. **Logging:** Replace `print()` statements with Python's `logging` module for configurable log levels.

3. **Error Handling:** Add more robust exception handling, especially for file I/O operations and external library calls.

4. **Testing:** Expand test coverage beyond the current 25 unit tests. Add integration tests for the full pipeline.

### Performance

1. **Multi-threading:** The demo already has threads for local mapping and loop detection. Consider thread pools for feature detection/matching.

2. **GPU Acceleration:** Consider CUDA-accelerated feature detection (opencv-contrib with CUDA) for real-time performance.

3. **Memory Management:** Implement map point garbage collection for long-running sessions.

### Architecture

1. **Plugin System:** Make detector, matcher, and optimizer pluggable for easy experimentation.

2. **ROS Integration:** Add ROS node wrappers for integration with robot systems.

3. **Configuration Files:** Support YAML/JSON configuration files in addition to programmatic config.