# Requirements Document

# Monocular SLAM Trajectory Drift Bugfix Requirements

## Introduction

Monocular SLAM trajectory drifts continuously during long sequences due to scale ambiguity and insufficient loop closure constraints. On KITTI sequence 00, the system currently achieves 197m Absolute Pose Error (APE) instead of the target <20m. This document specifies requirements to address this issue through robust scale estimation and strengthened loop detection.

## Overview

Monocular SLAM trajectory drifts continuously during long sequences due to scale ambiguity and insufficient loop closure constraints. On KITTI sequence 00, the system currently achieves 197m Absolute Pose Error (APE) instead of the target <20m. Two primary issues contribute to this drift:

1. **Scale Estimation Noise**: The median depth filtering in `_recover_scale()` uses a simple median of recent map points without outlier rejection, causing spurious depth estimates to corrupt scale recovery.
2. **Insufficient Loop Detection**: Loop detector parameters are conservative, missing valid loop closures that would provide global constraints to correct accumulated drift.

This bugfix addresses both issues through:
- **Statistical Outlier Rejection**: Filter out depth outliers using IQR (Interquartile Range) based methodology before computing median.
- **Strengthened Loop Detection**: Reduce temporal dead zone and lower BoW score threshold to catch more valid loop closures.

---

## Bug Description

### Glossary

- **APE (Absolute Pose Error)**: Difference between estimated and ground truth camera poses
- **Scale Recovery**: Process of estimating monocular scale from map point depths
- **IQR (Interquartile Range)**: Q3 - Q1 where Q1, Q3 are 25th and 75th percentiles
- **Loop Closure**: Detection of revisited locations in the scene for global optimization
- **BoW (Bag of Words)**: Visual vocabulary-based place recognition
- **Outlier**: Depth value outside the central distribution, indicating measurement error or triangulation failure
- **Geometric Verification**: PnP RANSAC confirmation of loop closure validity

### Current Observed Behavior

- **Symptom**: Continuous trajectory drift accumulating throughout KITTI sequence 00
- **Current APE**: ~197m (measured on full sequence)
- **Manifestation**: Error increases monotonically with sequence length; camera poses diverge from ground truth despite successful local feature tracking and bundle adjustment
- **Affected Components**: Scale recovery (`_recover_scale()`) and loop detection (`LoopDetector`)

### When the Bug Occurs

The bug manifests under the following conditions:

1. **Scale Estimation Bug**: When computing scale in `_recover_scale()` with `scale_mode='median_depth'`
   - Recent map points contain outliers (points with incorrect or extreme depth values due to triangulation errors, matches in untextured regions, or tracking failures)
   - These outliers corrupt the median depth estimate, producing incorrect scale factors (either too large or too small)
   - Scale factor is clamped to [0.3m, 80.0m] but this is insufficient to reject pathological outliers

2. **Loop Detection Bug**: During long sequences where loop closures occur
   - Loop detector has `min_loop_gap_frames=200`, which suppresses potential loop closures that occur within 200 frames of the previous verification
   - `min_bow_score=0.012` is too conservative, rejecting valid loop candidates with marginally lower scores
   - Many valid loop closure opportunities are missed, preventing global optimization from correcting drift

### Expected Correct Behavior

**Target APE**: Reduce to <20m on KITTI sequence 00

When implemented correctly:
1. **Scale Estimation**: Depths are filtered for outliers (IQR method) before median computation, producing robust scale estimates
2. **Loop Detection**: Parameters are tuned to detect valid loop closures more frequently, providing global constraints that correct accumulated drift through pose graph optimization

---

## Requirements

### 1. Outlier Rejection in Scale Recovery

**Requirement 1.1 - IQR-Based Outlier Detection**

When computing median depth in `_recover_scale()` for `scale_mode='median_depth'`:
- Extract depth values from recent map points
- Compute Interquartile Range (IQR): `IQR = Q3 - Q1` where Q1, Q3 are 25th and 75th percentiles
- Define outlier bounds: `lower = Q1 - 1.5 × IQR`, `upper = Q3 + 1.5 × IQR` (Tukey fences)
- Retain only depths within `[lower, upper]` bounds
- Compute median of filtered depths

**Validates**: The bug condition where outliers corrupt depth estimates

**Rationale**: IQR-based filtering (Tukey fences) is a standard statistical method for robust outlier detection that:
- Preserves the median as the central tendency measure
- Automatically adapts to the distribution of depths in the recent window
- Removes extreme outliers while retaining inliers near the true depth

**Example Input/Output**:
- Input: Recent depths = [2.5, 3.1, 3.2, 3.0, 150.0, 2.9, 3.1] (150.0 is outlier from failed triangulation)
- Q1 ≈ 2.95, Q3 ≈ 3.15, IQR ≈ 0.2
- Bounds: lower ≈ 2.65, upper ≈ 3.45
- Filtered depths: [2.5, 3.1, 3.2, 3.0, 2.9, 3.1]
- Median: 3.05 (robust estimate)

**Requirement 1.2 - Edge Case Handling**

- **Few map points** (< 5): Return 1.0 (no scale correction available)
- **All points filtered as outliers**: Return 1.0 (insufficient inliers)
- **Non-finite depths** (NaN, inf): Filter out before IQR computation
- **Negative depths**: Filter out (impossible in physical camera frame)

**Rationale**: Edge cases must be handled gracefully to prevent crashes and maintain fallback behavior

**Requirement 1.3 - Preservation of Diagnostic Modes**

- `scale_mode='gt'`: Continue deriving scale from ground truth poses (unchanged)
- `scale_mode='fixed'`: Continue using `cfg.fixed_scale` (unchanged)
- `scale_mode='none'`: Continue returning 1.0 (unchanged)
- Only `scale_mode='median_depth'` applies outlier filtering

**Rationale**: Diagnostic modes must remain unchanged to allow validation of other SLAM components independently

**Requirement 1.4 - Clamping After Filtering**

After computing filtered median:
- Apply existing clamp to `[scale_clamp_min, scale_clamp_max]` = `[0.3m, 80.0m]`
- This secondary safety check ensures scale stays within physical bounds

**Rationale**: Preserves existing safety mechanism as a last resort

---

### 2. Strengthened Loop Detection

**Requirement 2.1 - Reduced Temporal Dead Zone**

In `LoopDetector.__init__()`:
- Change `min_loop_gap_frames` from 200 to **100 frames** (≈3.33 seconds at 30 Hz)
- Dead zone now allows loop closures to fire more frequently

**Rationale**: 
- Current value (200 frames) is too conservative for KITTI (each sequence ~130 seconds)
- Reducing to 100 allows legitimate revisits to be detected without flooding PGO
- Still provides safety margin to prevent extraneous duplicates

**Example**: 
- Current: If loop verified at frame 500, next loop suppressed until frame 700
- New: If loop verified at frame 500, next loop can fire at frame 600 (earlier revisit of different location)

**Requirement 2.2 - Lowered BoW Score Threshold**

In `LoopDetector.__init__()`:
- Change `min_bow_score` from 0.012 to **0.009**
- Lower threshold captures more valid place recognition candidates

**Rationale**:
- Current threshold (0.012) requires very high descriptor similarity, rejecting marginal matches
- Lowering to 0.009 captures loop candidates with good (but not perfect) visual similarity
- Geometric verification (PnP RANSAC) still enforces strict inlier requirements, preventing false positives

**Example**:
- A revisited loop with slight camera angle variation might score 0.010 (currently rejected, now accepted)
- Geometric verification ensures only geometrically consistent matches produce events

**Requirement 2.3 - Preservation of Existing Loop Detection Behavior**

- `min_geo_inliers`: Remain at 15 (current setting, sufficient for geometric verification)
- `consistency`: Remain at 3 (consecutive BoW hits required)
- `temporal_window`: Remain at 20 (exclusion zone for recent keyframes)
- Geometric verification (PnP RANSAC): No changes to reprojection error threshold (8.0px)
- Callback exception handling: Preserved (thread continues on callback exceptions)
- Match-KF deduplication (FIX 2): Preserved (prevent same landmark from triggering multiple events)

**Rationale**: These parameters work well; only temporal dead zone and BoW threshold are too conservative

**Requirement 2.4 - Parameter Tunability**

Loop detector parameters MUST be tunable at runtime via `LoopDetector.__init__()` and `VOConfig`:
- `min_bow_score`: configurable (default 0.009)
- `min_loop_gap_frames`: configurable (default 100)
- `min_geo_inliers`: configurable (default 15)
- `consistency`: configurable (default 3)
- `temporal_window`: configurable (default 20)

**Rationale**: Allows experiments with different parameter values without code changes

---

### 3. Integration Requirements

**Requirement 3.1 - Compatibility with Local Bundle Adjustment**

The scale recovery output must work correctly with Local BA:
- Scaled poses are stored in `self.T_world_cam`
- BA uses these poses as initial estimates
- BA refines poses without re-scaling

**Test**: Run Local BA on frames with and without scale recovery; verify BA converges without numerical issues

**Requirement 3.2 - Compatibility with Pose Graph Optimization**

Loop closure events trigger PGO with relative transforms from `LoopEvent.T_rel`:
- Loop events must contain geometrically valid transforms
- Increased loop event frequency must not overwhelm PGO solver (dead zone prevents this)

**Test**: Verify PGO converges with increased loop frequency; check optimization residuals

**Requirement 3.3 - Compatibility with Feature Tracking**

The fix must not affect feature extraction, matching, or tracking:
- `features.py`, `tracking()` in `pipeline.py`: unchanged
- Keypoint detection, description: unchanged
- Feature matching: unchanged

**Test**: Verify feature counts and tracking success rates unchanged

**Requirement 3.4 - Fallback Behavior Preservation**

- If `map_points` is empty: return 1.0 (no change)
- If pose is non-finite: return 1.0 (no change)
- If all depths filtered as outliers: return 1.0 (no change)
- Return value must always be `> 0.0` and finite

**Rationale**: Maintains robustness when data is insufficient

---

### 4. Validation Requirements

**Requirement 4.1 - Fix Checking on Buggy Inputs**

For sequences with trajectory drift:
- Measure APE before fix: ~197m
- Measure APE after both fixes: target <20m
- Improvement: ≥177m reduction

**Test Dataset**: KITTI sequence 00 (4541 frames, known ground truth)

**Requirement 4.2 - Preservation Checking**

For all non-buggy inputs, behavior must be unchanged:
- Diagnostic modes (`scale_mode='gt'`, `'fixed'`, `'none'`): produce identical results
- Single keyframe scenarios: return 1.0 (unchanged)
- Loop detection in scenes without loops: no change
- Feature tracking: no change

**Test Approach**: Run diagnostic mode tests; compare statistics with baseline

**Requirement 4.3 - Edge Case Validation**

- **Few map points** (< 5): handle gracefully, return 1.0
- **All outliers**: handle gracefully, return 1.0
- **Non-finite depths**: filter without crashing
- **Negative depths**: filter without crashing

**Test**: Unit tests for each edge case

---

## Acceptance Criteria

### AC 2.1 - Outlier Rejection Implementation

**Given** a sequence of monocular SLAM frames with ground truth poses  
**When** `_recover_scale()` is called with `scale_mode='median_depth'` and recent map points contain outliers  
**Then** the function SHALL filter depths using IQR-based outlier detection (Tukey fences with multiplier 1.5) before computing median  
**And** the returned scale factor SHALL be within `[scale_clamp_min, scale_clamp_max]`  
**And** edge cases (few points, all outliers, non-finite depths) SHALL return 1.0  

**Validates**: Requirements 1.1, 1.2, 1.4

### AC 2.2 - Loop Detection Parameter Tuning

**Given** the KITTI sequence 00 with multiple loop closure opportunities  
**When** `LoopDetector` is initialized with `min_loop_gap_frames=100` and `min_bow_score=0.009`  
**Then** the detector SHALL verify more loop closures than with default parameters (min_loop_gap_frames=200, min_bow_score=0.012)  
**And** the increased loop events SHALL be geometrically verified (PnP RANSAC with ≥15 inliers)  
**And** no false loop closures SHALL be accepted  

**Validates**: Requirements 2.1, 2.2, 2.3

### AC 2.3 - Preservation of Diagnostic Modes

**Given** `scale_mode` set to `'gt'`, `'fixed'`, or `'none'`  
**When** `_recover_scale()` is called  
**Then** the function SHALL produce identical behavior as before the fix  
**And** diagnostic mode isolation SHALL allow independent validation of scale recovery vs. other SLAM components  

**Validates**: Requirement 1.3

### AC 2.4 - Trajectory Drift Reduction

**Given** KITTI sequence 00 with current APE ~197m  
**When** both median depth filtering (with outlier rejection) AND strengthened loop detection are applied  
**Then** the final APE SHALL reduce to <20m  
**And** the trajectory SHALL match ground truth within visual inspection accuracy  

**Validates**: Requirements 1.1, 2.1, 2.2 (composite acceptance)

### AC 2.5 - Integration Compatibility

**Given** the fixed `_recover_scale()` and `LoopDetector` integrated into the full SLAM pipeline  
**When** running on KITTI sequence 00  
**Then** Local BA SHALL converge without numerical issues  
**And** PGO SHALL optimize with increased loop frequency without divergence  
**And** feature tracking statistics SHALL remain unchanged  
**And** diagnostic modes SHALL work as before  

**Validates**: Requirements 3.1, 3.2, 3.3, 3.4

---

## Out of Scope

The following are explicitly NOT addressed by this bugfix:

- **Camera calibration refinement**: Intrinsic parameters are fixed
- **Feature detector/matcher improvements**: ORB-SLAM2 features used as-is
- **Initial pose estimation**: Initialization strategy unchanged
- **Bundle adjustment algorithm**: Local BA and PGO algorithms unchanged
- **Keyframe selection criteria**: Keyframe policy unchanged
- **Map point culling strategy**: Culling rules unchanged
- **Real-time performance optimization**: Only correctness is targeted
- **Other KITTI sequences**: Focus is on sequence 00; other sequences may require different tuning

---

## Dependencies and Constraints

- **Python version**: 3.8+ (numpy, cv2, scipy available)
- **External libraries**: OpenCV (cv2), NumPy, SciPy
- **Backward compatibility**: Must preserve existing API; only internal implementation of `_recover_scale()` changes
- **No breaking changes**: All existing tests and diagnostic modes must continue to work

---

## Metrics for Success

1. **APE Reduction**: From ~197m to <20m on KITTI sequence 00
2. **Loop Events**: Increase from current baseline to >50 verified loops (example target)
3. **Outlier Rejection Effectiveness**: >10% of recent depths filtered as outliers (indicating filter is working)
4. **No Regressions**: Diagnostic modes produce identical results; feature tracking unchanged
5. **Edge Case Robustness**: All edge cases handled gracefully without crashes
