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
tests/test_vo.py       — Unit tests (pytest)
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Synthetic flythrough (no data needed)
python demo.py

# KITTI dataset  (download sequence 00 from http://www.cvlibs.net/datasets/kitti/)
python demo.py --kitti /data/kitti/sequences/00/image_0

# Video file
python demo.py --video my_drive.mp4

# Webcam
python demo.py --webcam

# Headless (no GUI, just save trajectory.png)
python demo.py --no-gui --output traj.png
```

---

## KITTI Setup

1. Download odometry dataset: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
2. Extract `data_odometry_gray.zip` — gives `sequences/00/image_0/` etc.
3. Camera 0 intrinsics for seq 00 are already built into `CameraModel.kitti()`.
4. Ground truth poses are in `poses/00.txt` (4×4 row-major SE3 lines).

---

## Programmatic Usage

```python
from vo import CameraModel, VisualOdometry, VOConfig, DetectorType

# Build camera
camera = CameraModel.kitti()

# Optionally customise
cfg = VOConfig(
    detector_type  = DetectorType.ORB,
    max_features   = 1500,
    ratio_thresh   = 0.75,
    min_inliers    = 20,
    scale_mode     = "median_depth",  # monocular scale heuristic
)

vo = VisualOdometry(camera, cfg)

for frame in my_frame_source:
    stats = vo.process(frame)

    # Current camera pose in world frame (4×4 SE3)
    T = vo.T_world_cam

    # All accumulated poses
    traj = vo.trajectory       # (N, 3)  camera centres

    # Map points for local BA
    mps  = vo.map_points       # List[MapPoint]

    # Keyframes
    kfs  = vo.keyframes        # List[Keyframe]
```

---

## VSLAM Extension Points

| Hook | How to use |
|------|-----------|
| **New keyframe callback** | `vo.on_new_keyframe = my_loop_closure_fn` |
| **Pose correction** | `vo.pose_graph.update(frame_id, corrected_T)` |
| **Map points** | `vo.map_points` — feed to local bundle adjustment |
| **Keyframes** | `vo.keyframes` — DBoW2 place recognition, co-visibility graph |
| **PoseGraph** | `vo.pose_graph` — add edges, run pose-graph optimization |

### Suggested additions to grow into full VSLAM

1. **Loop Detection** — DBoW2 / NetVLAD on keyframe descriptors
2. **Loop Correction** — Sim3 alignment + pose-graph optimization (g2o / GTSAM)
3. **Local Bundle Adjustment** — optimize recent KF poses + map points (g2o)
4. **Map Culling** — prune redundant keyframes (>90% overlapping features)
5. **Relocalization** — match current frame to all keyframes via BoW when LOST

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
| `kf_max_frames` | `20` | Force KF after this many frames |
| `max_reproj_err` | `2.0 px` | Triangulation filter threshold |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Coordinate Convention

- **Camera frame**: OpenCV (+x right, +y down, +z into scene)
- **World frame**: set to identity at frame 0
- **Pose**: `T_world_cam` (4×4 SE3) — transforms camera → world
- **Translation**: unit-norm in monocular mode; metric if scale is known
