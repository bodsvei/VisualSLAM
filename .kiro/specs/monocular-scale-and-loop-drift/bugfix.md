# Monocular SLAM Trajectory Drift Bugfix Requirements

## Introduction

A monocular Visual SLAM system is experiencing severe trajectory drift during KITTI sequence evaluation, with Relative Pose Error (RPE) errors of 150-500m for 100m deltas and Absolute Pose Error (APE) errors of 50-350m throughout the sequence. The root cause stems from two compounding issues:

1. **Weak Median Depth Filtering**: The scale recovery mechanism uses the median of recent map point depths but lacks robust statistical outlier rejection. Erroneous scale estimates accumulate frame-to-frame, compounding into substantial translation errors.

2. **Insufficient Loop Closure Detection**: The loop detector has high detection thresholds (min_bow_score=0.012, min_geo_inliers=30) and a wide temporal dead zone (min_loop_gap=200 frames), allowing loop closure opportunities to be missed. Detected loops are insufficient to correct drift before it compounds into the middle section of the trajectory.

**Impact**: System operates in diagnostic mode (scale_mode='gt' using ground truth scale) because the scale recovery is too unreliable for production. Even with all 4 processing stages enabled (Local BA, Loop Detection, Pose Graph Optimization, Delta Propagation), the system cannot adequately correct accumulated drift.

**Goal**: Fix both issues to reduce drift below 100m APE for KITTI sequence 00 by improving depth filtering robustness and increasing loop closure frequency through refined parameters.


## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN processing a monocular video sequence where estimated depth varies significantly frame-to-frame due to scale ambiguity THEN the system computes scale via median of recent map point depths without outlier rejection, allowing outlier-driven scale estimates to accumulate and compound into large translation errors

1.2 WHEN the loop detector identifies a potential loop closure candidate via Bag-of-Words matching THEN the system suppresses loop closure callbacks for 200 consecutive frames (min_loop_gap=200), preventing frequent loop corrections and allowing drift to compound before the next correction opportunity

1.3 WHEN a loop closure is detected and the system enters the temporal dead zone (min_loop_gap=200 frames) THEN subsequent loop detections are suppressed even if valid, missing correction opportunities during high-drift periods

1.4 WHEN map point depth values include outliers (e.g., erroneous triangulations, poorly-conditioned baselines) THEN the median is still influenced by outlier-skewed distributions, producing noisy scale estimates that fail to stabilize the trajectory


### Expected Behavior (Correct)

2.1 WHEN processing a monocular video sequence where estimated depth varies significantly frame-to-frame THEN the system SHALL compute scale via statistical outlier-robust depth filtering (e.g., IQR-based rejection, Z-score filtering, or MAD-based detection) before taking the median, producing stable and accurate scale estimates frame-to-frame

2.2 WHEN the loop detector identifies a valid loop closure THEN the system SHALL allow loop closure callbacks to fire with a refined temporal dead zone (min_loop_gap ≤ 100 frames, or adaptive to sequence speed), enabling more frequent corrections to accumulating drift

2.3 WHEN a loop closure correction is applied THEN the temporal dead zone SHALL resume, but the system SHALL NOT suppress additional loop closures if they occur outside the dead zone, enabling opportunistic re-correction during drift-sensitive periods

2.4 WHEN map point depth values include outliers THEN the system SHALL apply robust statistical outlier detection (e.g., IQR-based Tukey fences, MAD scoring, or median filtering) to reject outliers before computing the median, producing outlier-resistant scale estimates


### Unchanged Behavior (Regression Prevention)

3.1 WHEN the system processes frames with insufficient matches or broken ego-motion estimates THEN the system SHALL CONTINUE TO guard against invalid poses and return a fallback scale of 1.0

3.2 WHEN the system operates in scale_mode='gt' (diagnostic mode) THEN the system SHALL CONTINUE TO derive scale from ground truth poses without modification, preserving diagnostic accuracy

3.3 WHEN the system operates in scale_mode='fixed' or scale_mode='none' THEN the system SHALL CONTINUE TO return the fixed or unit scale without modification

3.4 WHEN the local bundle adjustment and pose graph optimization stages process keyframes THEN the system SHALL CONTINUE TO operate without modification, preserving trajectory refinement through geometric optimization

3.5 WHEN the loop detector processes frames outside detected loop closures THEN the system SHALL CONTINUE TO extract and track features normally, preserving visual tracking behavior

3.6 WHEN map points with normal depth distributions (within reasonable parallax and depth bounds) are observed THEN the system SHALL CONTINUE TO produce correct scale estimates after outlier filtering is applied

3.7 WHEN loop closures occur with sufficient temporal separation (outside the dead zone) THEN the system SHALL CONTINUE TO fire loop closure callbacks and trigger Pose Graph Optimization as before
