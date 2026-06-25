# GEMINI.md — vo_slam project context

> Gemini CLI reads this file automatically at session start.
> It gives the model persistent knowledge of the project so you
> don't have to re-explain context every session.

---

## Project overview

Python stereo visual-odometry / SLAM system evaluated on **KITTI sequence 00**.
The pipeline estimates camera trajectory from stereo image pairs and compares it
against ground-truth poses using the `evo` evaluation library.

Current status (June 2026):
- APE RMSE: **13.9 m** (down from 86.94 m)
- Sim(3) scale factor: **0.671×** — stereo baseline mis-calibrated (~49% too large)
- Pitch residual: **−31°** — stereo rectification tilt leaking into extrinsics
- Peak drift at **frame 4360** — loop closure not correcting late-sequence

---

## Repository layout

```
vo_slam/
├── pipeline.py              # Main entry point — chains all modules
├── motion.py                # Relative pose estimation (Essential matrix + recoverPose)
├── features.py              # ORB/SIFT detection, matching, optical flow
├── triangulation.py         # MapPoint creation from stereo/multi-view
├── keyframe.py              # Keyframe data structure
├── local_mapping.py         # Local BA thread, covisibility graph maintenance
├── bundle_adjustment.py     # g2o-based local/global bundle adjustment
├── covisibility.py          # Covisibility graph (KF–KF shared map points)
├── loop_detector.py         # Place recognition + geometric verification
├── pose_graph_optimizer.py  # g2o pose graph / scipy fallback
├── place_recognition.py     # BoW / descriptor-based place retrieval
├── relocalization.py        # Lost-tracking recovery
├── map_culling.py           # Remove redundant map points and keyframes
├── map_storage.py           # Serialise/load map to disk
├── stereo_calibration.py    # Offline stereo calibration script
├── camera.py                # CameraModel (K, baseline, distortion)
├── bow_database.py          # Bag-of-Words database wrapper
├── vocabulary.py            # ORB vocabulary loader
├── global_bundle_adapter.py # Adapter for full-map BA after loop closure
├── visualization.py         # Open3D / matplotlib trajectory/map display
└── test_vo.py               # Integration test runner
```

---

## Known bugs

- Geometry and pose accumulation bugs (Bugs 1-5) were fixed.
- Tracking logic refactored to use PnP fallback and CV motion model.

---

## Calibration status

| Parameter | Expected | Actual (calibration file) | Action needed |
|-----------|----------|--------------------------|---------------|
| Scale | 1.0× | 0.671× | Re-run stereoCalibrate, measure baseline physically |
| Pitch | 0° | −31° | Check R1[2,0]; compose R1 into camera extrinsic |
| Stereo RMS | < 0.5 px | unknown | Verify after re-calibration |

Check current calibration:
```python
import numpy as np
cal = np.load('stereo_calibration.npz')
print('Baseline (mm):', np.linalg.norm(cal['T']))
print('R1 pitch component:', cal['R1'][2, 0])   # should be ~0
print('Stereo RMS:', float(cal['rms']))
```

---

## Error metric history

| Run | APE RMSE | Median | Scale | Pitch | Notes |
|-----|----------|--------|-------|-------|-------|
| Monocular | 128.7 m | 115.0 m | 2.57× | −25.5° | Baseline |
| Stereo v1 | 82.7 m | 48.8 m | 0.68× | −29.6° | Scale flipped |
| Stereo v2 | 86.9 m | 51.8 m | 0.67× | −31.1° | Rotation fixes applied |
| Stereo v3 | 13.9 m | -      | -     | -      | Bugs 1-5 fixed + PnP fallback |

APE evaluated with `evo_ape` using Sim(3) Umeyama alignment on KITTI seq 00 (4541 frames).

---

## Development commands

```bash
# Run pipeline on KITTI seq 00
python -m vo_slam.pipeline --sequence 00 --dataset /data/KITTI

# Evaluate with evo
evo_ape kitti KITTI/poses/00.txt results_00.txt --align --plot --save_results ate.zip

evo_rpe kitti KITTI/poses/00.txt results_00.txt --align --delta 100 --delta_unit m --plot

# Stereo re-calibration
python -m vo_slam.stereo_calibration \
  --left  ./calib_images/left  \
  --right ./calib_images/right \
  --rows 9 --cols 13 --square 20.0

# Run tests
pytest vo_slam/test_vo.py -v
```

---

## Conventions (critical — read before touching pose code)

- **T_world_cam**: 4×4 SE3, transforms a point FROM camera frame INTO world frame.
  `p_world = T_world_cam @ p_cam`
- **T_cam_world**: inverse of above. What g2o `VertexSE3Expmap` stores internally.
- **recoverPose output**: T_{ref←cur} — must be inverted before world accumulation.
- **compose_pose(T1, T2)**: returns T1 @ T2 (world_ref @ ref_cur = world_cur).
- **invert_pose(T)**: uses R.T and -R.T @ t — do NOT use np.linalg.inv for SE3.

---

## Next priorities

1. **Re-calibrate stereo baseline** — fix the 0.671× scale (biggest remaining gain)
2. **Compose R1 into camera extrinsic** — fix the −31° pitch residual
3. **Add loop closure logging** — count candidates at frames 3500–4541 to debug Q4 drift