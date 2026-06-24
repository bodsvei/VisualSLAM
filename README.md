# VisualSLAM: Stereo Visual Odometry & SLAM in Python

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Status: Active Development](https://img.shields.io/badge/status-active_development-orange.svg)
![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)

A modular **Stereo Visual Odometry and SLAM** pipeline designed as a foundation for a full SLAM system. Built entirely in Python, this project aims for **StellaVSLAM-level** capability, progressively achieving local mapping, place recognition, and full loop closure.

Currently, the pipeline processes stereo image pairs to estimate camera trajectories and evaluates against ground-truth poses (e.g., KITTI sequence 00).

---

## Key Features

- **Stereo Front-End:** Uses stereo image pairs for robust metric scale estimation (transitioned from monocular VO).
- **Feature Tracking:** Support for ORB features with grid-based suppression and Brute-Force/FLANN matching.
- **Motion Estimation:** Essential matrix-based pose recovery with RANSAC.
- **Local Mapping & Bundle Adjustment:** Covisibility graph maintenance and `g2o`-backed local bundle adjustment to minimize drift.
- **Loop Detection & Place Recognition:** Bag-of-Words (BoW) vocabulary tree integration for robust loop candidate detection.
- **Evaluation Integration:** Built-in benchmarking against the KITTI dataset using the `evo` evaluation library.

---

## Current Performance

Evaluated on **KITTI Sequence 00**:
- **APE RMSE:** 13.92 m (Significant improvement from 86.94 m previous baseline)
- **Scale Factor:** 1.027× (Calibration successfully tuned)
- **Pitch Residual:** −0.45° 

---

## Installation & Quick Start

### Prerequisites
- macOS (Apple Silicon supported) / Linux
- Python 3.9+
- `g2o-python` for bundle adjustment

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/VisualSLAM.git
cd VisualSLAM

# Install editably to ensure proper namespace resolution
pip install -e .

# Install dependencies
pip install opencv-python numpy matplotlib pytest evo pyqt5
```

### Running the System

```bash
# Run unit tests
pytest vo_slam/test_vo.py -v

# Synthetic flythrough (no data needed)
python demo.py

# Headless mode (save trajectory to file without GUI)
python demo.py --no-gui --output traj.png

# Offline Stereo Calibration
python -m vo_slam.stereo_calibration --left ./calib_images/left --right ./calib_images/right --rows 9 --cols 13 --square 20.0
```

---

## Testing on KITTI

1. **Download the Dataset:**
   Register at [CVLIBS](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and download `data_odometry_gray.zip`, `data_odometry_poses.zip`, and `data_odometry_calib.zip`. Extract to a `KITTI/` directory.

2. **Run the Pipeline:**
   ```bash
   python -m vo_slam.pipeline --sequence 00 --dataset /path/to/KITTI
   ```

3. **Evaluate with `evo`:**
   ```bash
   # Absolute Trajectory Error (ATE)
   evo_ape kitti KITTI/poses/00.txt results_00.txt --align --plot --save_results ate.zip
   
   # Relative Pose Error (RPE)
   evo_rpe kitti KITTI/poses/00.txt results_00.txt --align --delta 100 --delta_unit m --plot
   ```
   > **macOS Tip:** If `--plot` hangs without a Qt backend, use `--save_plot ate.pdf` instead.

---

## Architecture

The pipeline is organized logically to separate the front-end tracking, mapping, and optimization components:

```
VisualSLAM/
├── vo_slam/                     # Core SLAM package
│   ├── pipeline.py              # Orchestrator & entry point
│   ├── motion.py                # Motion estimation (Essential matrix → R, t)
│   ├── features.py              # ORB/SIFT tracking & matching
│   ├── triangulation.py         # Stereo/Multi-view map point triangulation
│   ├── keyframe.py              # Keyframe data structure
│   ├── local_mapping.py         # Local BA thread & covisibility maintenance
│   ├── bundle_adjustment.py     # g2o-based local/global bundle adjustment
│   ├── covisibility.py          # Covisibility graph
│   ├── loop_detector.py         # Place recognition & geometric verification
│   ├── pose_graph_optimizer.py  # g2o pose graph optimization
│   ├── place_recognition.py     # BoW / descriptor-based place retrieval
│   ├── relocalization.py        # Lost-tracking recovery
│   ├── map_culling.py           # Remove redundant map points and keyframes
│   ├── map_storage.py           # Serialize/load map to disk
│   ├── stereo_calibration.py    # Offline stereo calibration script
│   ├── camera.py                # Camera intrinsics, extrinsics & distortion
│   ├── bow_database.py          # Bag-of-Words database wrapper
│   ├── vocabulary.py            # ORB vocabulary loader
│   ├── global_bundle_adapter.py # Adapter for full-map BA
│   ├── visualization.py         # Trajectory & feature mapping tools
│   └── test_vo.py               # Integration test runner
├── demo.py                      # Standalone runnable demo
├── diagnose.py                  # Debugging and diagnostic tools
└── run_kitti.py                 # KITTI evaluation script
```

---

## Active Development & Roadmap

VisualSLAM is being developed in iterative stages. The current focus is fully maturing the stereo SLAM pipeline:

- [x] **Stage 1:** Core Visual Odometry (Feature tracking, essential matrix, stereo triangulation)
- [x] **Stage 2:** Local Mapping (Covisibility graph, Local BA)
- [ ] **Stage 3:** Loop Closing (Sim3 solver, Pose graph optimization)
- [ ] **Stage 4:** Map Reuse & Relocalization (Safe persistence, BoW query matching)

### Current Priorities
1. **Stereo Calibration Tuning:** Fixing baseline miscalibration (scale & pitch residuals).
2. **Pose Accumulation Fixes:** Ensuring strict coordinate frame conventions (`T_world_cam`) during motion estimation and bundle adjustment read-backs.
3. **Loop Closure Robustness:** Stabilizing late-sequence loop closure logic (e.g., tracking drift around frame 4360).

---

## Security Notice

**Unsafe Deserialization:** The map serialization currently utilizes `pickle` (`map_storage.py`, `vocabulary.py`), which is susceptible to arbitrary code execution if untrusted map files are loaded. Avoid loading map/vocabulary files from untrusted sources until the system migrates to a secure serialization format (e.g., JSON + NumPy `.npz` or MessagePack).

---

## Contributing

Contributions are welcome! Please keep in mind the strict coordinate conventions used throughout the project:
- **`T_world_cam`:** 4×4 SE3 transforming from camera frame to world frame (`p_world = T_world_cam @ p_cam`).
- **Right-multiplication** is used for pose accumulation.
- Ensure any `cv2.recoverPose` output (`T_{ref←cur}`) is inverted before composing.