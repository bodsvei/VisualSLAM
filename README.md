# VisualSLAM: Stereo Visual Odometry & SLAM in Python

A modular **Stereo Visual Odometry and SLAM** pipeline designed as a foundation for a full SLAM system. Built entirely in Python, this project aims for **StellaVSLAM-level** capability, progressively achieving local mapping, place recognition, and full loop closure.

Currently, the pipeline processes stereo image pairs to estimate camera trajectories and evaluates against ground-truth poses (e.g., KITTI sequence 00).

## Current Performance

Evaluated on **KITTI Sequence 00**:
- **APE RMSE:** 13.9 m
- **Scale Factor:** 0.671× (stereo baseline mis-calibrated, ~49% too large)
- **Pitch Residual:** −31° (stereo rectification tilt leaking into extrinsics)

## Installation

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

## How to Use the Package

The main entry point for the SLAM pipeline is `vo_slam.pipeline`. It processes image sequences from datasets like KITTI.

### Running the System on KITTI

1. **Download the Dataset:**
   Register at [CVLIBS](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and download `data_odometry_gray.zip`, `data_odometry_poses.zip`, and `data_odometry_calib.zip`. Extract to a `KITTI/` directory.

2. **Run the Pipeline:**
   ```bash
   # Run pipeline on KITTI seq 00
   python -m vo_slam.pipeline --sequence 00 --dataset /path/to/KITTI
   ```

3. **Evaluate with `evo`:**
   ```bash
   # Absolute Trajectory Error (ATE)
   evo_ape kitti KITTI/poses/00.txt results_00.txt --align --plot --save_results ate.zip
   
   # Relative Pose Error (RPE)
   evo_rpe kitti KITTI/poses/00.txt results_00.txt --align --delta 100 --delta_unit m --plot
   ```

### Stereo Re-calibration

If you need to recalibrate the stereo camera:
```bash
python -m vo_slam.stereo_calibration \
  --left ./calib_images/left \
  --right ./calib_images/right \
  --rows 9 --cols 13 --square 20.0
```

### Running Tests

To run the integration tests:
```bash
pytest vo_slam/test_vo.py -v
```

## Architecture

The pipeline is organized logically to separate the front-end tracking, mapping, and optimization components:

- `vo_slam/pipeline.py`: Main entry point — chains all modules.
- `vo_slam/motion.py`: Relative pose estimation (Essential matrix + recoverPose).
- `vo_slam/features.py`: ORB/SIFT detection, matching, optical flow.
- `vo_slam/triangulation.py`: MapPoint creation from stereo/multi-view.
- `vo_slam/keyframe.py`: Keyframe data structure.
- `vo_slam/local_mapping.py`: Local BA thread, covisibility graph maintenance.
- `vo_slam/bundle_adjustment.py`: g2o-based local/global bundle adjustment.
- `vo_slam/covisibility.py`: Covisibility graph (KF–KF shared map points).
- `vo_slam/loop_detector.py`: Place recognition + geometric verification.
- `vo_slam/pose_graph_optimizer.py`: g2o pose graph / scipy fallback.

## Conventions

- **T_world_cam**: 4×4 SE3, transforms a point FROM camera frame INTO world frame (`p_world = T_world_cam @ p_cam`).
- **T_cam_world**: inverse of above. What g2o `VertexSE3Expmap` stores internally.
- **recoverPose output**: `T_{ref←cur}` — must be inverted before world accumulation.
- **compose_pose(T1, T2)**: returns `T1 @ T2` (world_ref @ ref_cur = world_cur).
- **invert_pose(T)**: uses `R.T` and `-R.T @ t` — do NOT use `np.linalg.inv` for SE3.