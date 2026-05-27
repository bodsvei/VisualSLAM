# Visual Odometry — Python

A modular monocular VO pipeline designed as a foundation for a full V-SLAM system.

---

## Architecture

```
vo/
├── camera.py          — CameraModel  (intrinsics, projection, back-projection)
├── features.py        — FeatureDetector + FeatureMatcher (ORB/SIFT, BF/FLANN)
├── motion.py          — MotionEstimator (Essential matrix → R, t via RANSAC)
├── triangulation.py   — Triangulator (DLT + reprojection filter → MapPoints)
├── keyframe.py        — Keyframe dataclass + KeyframeSelector policy
├── pipeline.py        — VisualOdometry orchestrator + PoseGraph
└── visualization.py   — FeatureOverlay, TrajectoryPlot, VOVisualizer
demo.py                — Runnable demo (synthetic / KITTI / video / webcam)
run_kitti.py           — Script for KITTI dataset evaluation
tests/test_vo.py       — Unit tests (pytest)
```

---

## Methodology

This repository implements a rigorous, classic feature-based monocular visual odometry system. 

1. **Feature Detection & Matching**: Uses ORB features by default with grid-based suppression to ensure uniformly distributed keypoints. Matching is performed using Brute-Force Hamming distance, aggressively filtered by Lowe's ratio test (0.75 threshold) and cross-checking.
2. **Motion Estimation**: Relative pose is estimated via the Essential Matrix (`cv2.findEssentialMat`) using RANSAC. The Essential Matrix is decomposed into Rotation ($R$) and translation ($t$) using `cv2.recoverPose`, which includes a built-in chirality check to ensure triangulated points lie in front of both cameras. 
3. **Scale Recovery**: Since monocular VO lacks metric scale, scale is recovered dynamically using a `median_depth` heuristic. It normalizes the translation vector so that newly triangulated map points maintain a median depth consistent with the last 200 observed map points.
4. **Pose Accumulation**: Poses are represented in the **Camera-to-World** convention (`T_world_cam`) adhering to OpenCV coordinates ($+Z$ forward, $+X$ right, $+Y$ down). Absolute poses are composed via right-multiplication ($T_{world\_cam\_new} = T_{world\_cam\_prev} \times T_{rel}$).
5. **Keyframe Selection**: To reduce drift and computational load, poses are estimated between the current frame and the last selected **Keyframe**, rather than strictly consecutive frames. Keyframes are selected based on parallax, feature matching ratios, and rotational thresholds.

---

## Quick Start

```bash
pip install -r requirements.txt

# Synthetic flythrough (no data needed)
python demo.py

# Video file
python demo.py --video my_drive.mp4

# Webcam
python demo.py --webcam

# Headless (no GUI, just save trajectory.png)
python demo.py --no-gui --output traj.png
```

---

## Testing on KITTI — Step by Step

### Step 1 — Download the dataset

Go to: **http://www.cvlibs.net/datasets/kitti/eval_odometry.php**

You need to register (free). Download `data_odometry_gray.zip`, `data_odometry_poses.zip`, and `data_odometry_calib.zip`. Extract them so your folder structure looks like:

```
kitti/
├── sequences/
│   ├── 00/
│   │   ├── image_0/        ← left camera (use this)
│   │   ├── calib.txt
│   │   └── times.txt
│   └── ... (00–10 have ground truth)
└── poses/
    ├── 00.txt
    └── ... (ground truth SE3 poses)
```

### Step 2 — Run the KITTI evaluation script

We have provided `run_kitti.py` to process the dataset and export poses. Update the `KITTI_ROOT` variable in the script to point to your dataset directory.

```bash
python3 run_kitti.py
```
This will output `results_00.txt`.

### Step 3 — Evaluate with `evo`

`evo` is the standard toolkit for evaluating VO/SLAM:
```bash
pip install evo
```

**ATE** (Absolute Trajectory Error):
```bash
evo_ape kitti kitti/poses/00.txt results_00.txt --plot --plot_mode xz
```

**RPE** (Relative Pose Error) — Local drift per 100m segment:
```bash
evo_rpe kitti kitti/poses/00.txt results_00.txt --delta 100 --delta_unit m --plot --plot_mode xz
```

Without Bundle Adjustment or Loop Closure, expect an ATE of ~50–150m on Sequence 00. The RPE will give you a better sense of local tracking accuracy.

---

## Roadmap: Your VO → StellaVSLAM Level

Here's where the project currently stands and what lies ahead:

```
[YOU ARE HERE]
     │
     ▼
Stage 1: Solid VO          ✅  Done
Stage 2: Local Mapping     🔲  Next
Stage 3: Place Recognition 🔲
Stage 4: Loop Closing      🔲
Stage 5: Map Reuse         🔲
Stage 6: Full SLAM System  🔲  = StellaVSLAM level
```

### Stage 1 — Solid Visual Odometry ✅
The foundational tracking pipeline is complete (ORB, Essential matrix, Triangulation, Pose graph). The next step is evaluating on KITTI to ensure local drift is bounded.

### Stage 2 — Local Mapping & Bundle Adjustment
This prevents short-term drift from accumulating. 
*   **Co-visibility Graph**: A graph mapping keyframes to shared map points.
*   **Local Bundle Adjustment**: Jointly optimize poses and map points for a local window of covisible keyframes using `g2o`.
*   **Map Culling**: Prune redundant keyframes and bad map points.

### Stage 3 — Place Recognition
Detects when the camera returns to a visited location using a **Bag of Words (BoW)** vocabulary tree (e.g., DBoW3 with ORBvoc.txt) to match the current frame against the database of keyframes.

### Stage 4 — Loop Closing
Corrects the accumulated drift upon a loop detection.
*   **Sim3 Solver**: Compute a 7-DOF transform (R, t, scale) to correct scale drift.
*   **Pose-graph Optimization**: Propagate the Sim3 correction through the essential graph to update all poses efficiently.

### Stage 5 — Map Reuse & Relocalization
*   **Relocalization**: If tracking is LOST, match the current frame against the BoW database and solve PnP to recover the pose.
*   **Map Save/Load**: Serialize the map, poses, and BoW database to disk for multi-session mapping.

---

## Programmatic Usage

```python
from vo_slam import CameraModel, VisualOdometry, VOConfig, DetectorType

# Build camera
camera = CameraModel.kitti()

# Optionally customise
cfg = VOConfig(
    detector_type  = DetectorType.ORB,
    max_features   = 1500,
    ratio_thresh   = 0.75,
    scale_mode     = "median_depth",
)

vo = VisualOdometry(camera, cfg)

for frame in my_frame_source:
    stats = vo.process(frame)
    T = vo.T_world_cam         # 4x4 SE3 Pose
    traj = vo.trajectory       # Accumulated positions (N, 3)
```

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detector_type` | `ORB` | ORB / SIFT / FAST_ORB |
| `max_features` | `2000` | Max keypoints per frame |
| `ratio_thresh` | `0.75` | Lowe's ratio test threshold |
| `min_inliers` | `20` | Min RANSAC inliers to accept pose |
| `scale_mode` | `median_depth` | Monocular scale: `median_depth` / `fixed` / `none` |
| `kf_min_parallax` | `2.0` | Min pixel displacement to trigger new KF |

---

## Coordinate Convention

- **Camera frame**: OpenCV (+x right, +y down, +z into scene)
- **World frame**: set to identity at frame 0
- **Pose**: `T_world_cam` (4×4 SE3) — transforms camera → world
- **Translation**: unit-norm in monocular mode; metric if scale is known
