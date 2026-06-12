# Requirements Document: Monocular SLAM 3D-to-2D PnP Tracking

## Introduction

This feature transitions the monocular SLAM system from 2D-to-2D frame-to-frame tracking (via Essential Matrix decomposition) to 3D-to-2D tracking (via Perspective-n-Point with RANSAC). The current 2D-to-2D approach suffers from systematic scale collapse during camera acceleration because Essential Matrix decomposition normalizes translation to unit norm (||t|| = 1), causing estimated depths to be inversely proportional to camera velocity. This architectural change addresses the root cause by matching 2D image features directly against persistent 3D map points, eliminating the normalized baseline problem entirely.

## Glossary

- **PnP (Perspective-n-Point)**: Pose estimation algorithm that computes camera pose from 2D-3D correspondences (image points matched to known 3D map points)
- **2D-to-2D Tracking**: Current approach using Essential Matrix decomposition from frame-to-frame feature matches (suffers from unit baseline normalization)
- **3D-to-2D Tracking**: Proposed approach matching 2D features against persistent 3D map points via PnP (metric scale preserved)
- **Essential Matrix**: Geometric constraint relating corresponding image points in stereo/sequential views, normalized to remove scale information
- **Scale Collapse**: Systematic underestimation of depth during camera acceleration due to normalized translation baseline
- **LocalMapper**: Component maintaining the persistent 3D map (collection of MapPoint objects with world coordinates)
- **MapPoint**: 3D point in world coordinates with descriptor and observation history
- **Triangulation**: Process of computing 3D coordinates from 2D correspondences across multiple views
- **Fallback Strategy**: Recovery mechanism when insufficient 3D map points exist (initialization phase, tracking loss)
- **Bundle Adjustment (BA)**: Joint optimization of camera poses and 3D point positions to minimize reprojection error
- **Pose Graph Optimization (PGO)**: Backend optimization using loop closure constraints to correct accumulated drift

---

## Bug Description

### Root Cause Analysis

The exploration test (`test_scale_acceleration_bug.py`) demonstrated that the **true root cause** of scale collapse is NOT insufficient outlier rejection, but rather the **normalized 2D-to-2D translation baseline** in Essential Matrix decomposition:

1. **Current Pipeline**: Uses `cv2.findEssentialMat()` followed by `cv2.recoverPose()` for frame-to-frame tracking
2. **Essential Matrix Property**: Decomposition normalizes translation to ||t|| = 1 (unit norm)
3. **Acceleration Scenario**: When camera velocity doubles, actual baseline doubles but recovered t remains unit
4. **Scale Inversion**: Depths estimated from fast motion are systematically HALF of those from slow motion
5. **Feedback Loop**: Incorrect scale corrupts pose estimates, which corrupt future triangulations, compounding error

### Test Evidence

From `test_scale_acceleration_bug.py`:

**Concrete Test**:
- Camera moves 1.0m (slow step, frames 0→1)
- Camera moves 2.0m (fast step, frames 1→2, 2× acceleration)
- **Expected**: Depth ratio ≈ 2.0 (fast baseline is 2× slow)
- **Actual (unfixed)**: Depth ratio ≈ 1.0 (NO difference despite 2× baseline)

**Parametrized Tests**:
- All acceleration factors (1.5×, 2×, 3×, 4×) exhibit systematic scale collapse
- Property-based test generates counterexamples confirming scale inversion

### Current Observed Behavior

- **Symptom**: Scale estimates fail to track true baseline changes during acceleration/deceleration
- **Manifestation**: APE ~197m on KITTI sequence 00 (target <20m)
- **Affected Component**: `VisualOdometry.tracking()` and `_recover_scale()` in `pipeline.py`

---

## Proposed Architectural Fix

**Transition to 3D-to-2D Tracking via PnP**:

1. **Match 2D Features Against 3D Map**: For each new frame, match detected 2D features directly against persistent 3D map points from LocalMapper
2. **PnP Pose Estimation**: Use `cv2.solvePnPRansac()` to compute camera pose in the map's metric coordinate frame
3. **Scale Consistency**: New triangulated points inherit metric scale from PnP-estimated poses (no normalized baseline)
4. **Fallback to 2D-to-2D**: When insufficient 3D map points exist (initialization, tracking loss), fall back to Essential Matrix method
5. **Backend Compatibility**: BA and PGO continue using the same map points and poses, but with correct metric scale

**Key Benefit**: PnP computes translation directly in the metric of existing map points, completely cutting the runaway scale feedback loop.

---

## Requirements

### Requirement 1: 3D Map Point Indexing and Querying

**User Story**: As a SLAM developer, I want efficient access to active 3D map points with their descriptors, so that I can match incoming 2D features against the persistent map.

#### Acceptance Criteria

1. THE LocalMapper SHALL maintain an index of active MapPoint objects with world coordinates and descriptors
2. WHEN queried for map points, THE LocalMapper SHALL return all map points observed in the last N keyframes (configurable, default N=10)
3. THE LocalMapper SHALL provide map points with associated descriptors for feature matching
4. WHEN a map point is culled, THE LocalMapper SHALL remove it from the active index
5. THE active map point query SHALL complete in O(N × M) where N=number of keyframes, M=average map points per keyframe

---

### Requirement 2: 2D-to-3D Feature Matching

**User Story**: As a SLAM developer, I want to match 2D features in the current frame against 3D map points, so that I can establish correspondences for PnP pose estimation.

#### Acceptance Criteria

1. WHEN a new frame is processed, THE System SHALL extract 2D features (keypoints and descriptors) using the configured detector
2. THE System SHALL retrieve active 3D map points from LocalMapper with their descriptors
3. THE System SHALL match 2D frame descriptors against 3D map point descriptors using the configured matcher (default: BFMatcher with Hamming distance for ORB)
4. THE System SHALL apply Lowe's ratio test (default ratio=0.75) to filter ambiguous matches
5. WHEN a 2D feature matches multiple 3D map points, THE System SHALL retain only the best match (minimum descriptor distance)
6. THE System SHALL return a list of 2D-3D correspondences: (2D image coordinate, 3D world coordinate) pairs

---

### Requirement 3: PnP Pose Estimation with RANSAC

**User Story**: As a SLAM developer, I want to estimate camera pose from 2D-3D correspondences using PnP with RANSAC, so that I can compute metric-scale pose without normalized baseline issues.

#### Acceptance Criteria

1. WHEN at least 6 2D-3D correspondences exist, THE System SHALL invoke `cv2.solvePnPRansac()` to estimate camera pose
2. THE PnP solver SHALL use the camera intrinsic matrix (K) and assume zero distortion coefficients
3. THE PnP solver SHALL use RANSAC with configurable parameters:
   - `reprojectionError`: maximum reprojection error in pixels (default: 4.0px)
   - `iterationsCount`: maximum RANSAC iterations (default: 300)
   - `confidence`: required confidence level (default: 0.99)
4. THE PnP solver SHALL return rotation vector (rvec), translation vector (tvec), and inlier mask
5. THE System SHALL convert rvec to rotation matrix using `cv2.Rodrigues()` and construct T_cam_world as SE(3) transformation
6. WHEN PnP inlier count is below a threshold (default: 15 inliers), THE System SHALL reject the pose estimate and trigger fallback
7. THE System SHALL compute reprojection error for inliers and log the result for diagnostics

---

### Requirement 4: Fallback to 2D-to-2D Tracking

**User Story**: As a SLAM developer, I want automatic fallback to Essential Matrix-based tracking when insufficient 3D map points exist, so that the system can initialize and recover from tracking loss.

#### Acceptance Criteria

1. WHEN the number of active 3D map points is below a threshold (default: 50 points), THE System SHALL use 2D-to-2D tracking via Essential Matrix decomposition
2. WHEN PnP fails to find sufficient inliers (< 15), THE System SHALL fall back to 2D-to-2D tracking for the current frame
3. WHEN in fallback mode, THE System SHALL use the existing `cv2.findEssentialMat()` and `cv2.recoverPose()` pipeline
4. WHEN in fallback mode, THE System SHALL triangulate new map points using the fallback pose estimate
5. THE System SHALL log a warning when fallback is triggered, indicating reason (initialization, tracking loss, insufficient inliers)
6. WHEN sufficient 3D map points accumulate (≥ 50 points), THE System SHALL automatically transition back to PnP tracking

---

### Requirement 5: Scale Consistency in Triangulation

**User Story**: As a SLAM developer, I want newly triangulated map points to use metric scale from PnP-estimated poses, so that scale remains consistent throughout the map.

#### Acceptance Criteria

1. WHEN triangulating new map points, THE Triangulator SHALL use camera poses (T_world_cam) estimated by PnP
2. THE triangulated 3D coordinates SHALL be in the same world coordinate frame as existing map points
3. WHEN in fallback mode (2D-to-2D tracking), THE System SHALL apply scale correction to align the fallback pose with existing map scale before triangulation
4. THE scale correction SHALL compute median depth of existing map points visible in the current frame and scale the fallback translation accordingly
5. FOR ALL newly triangulated map points, the depth values SHALL be consistent with existing map points (within ±20% for points at similar world locations)

---

### Requirement 6: Performance Requirements

**User Story**: As a SLAM developer, I want PnP tracking to maintain real-time performance, so that the system remains practical for online SLAM applications.

#### Acceptance Criteria

1. THE 2D-to-3D matching step SHALL complete in <10ms for up to 500 2D features and 1000 3D map points
2. THE PnP RANSAC estimation SHALL complete in <15ms with default parameters (300 iterations, confidence=0.99)
3. THE total tracking overhead (matching + PnP) SHALL be <30ms per frame, measured on reference hardware (modern CPU, no GPU acceleration)
4. WHEN PnP tracking overhead exceeds 50ms, THE System SHALL log a performance warning
5. THE System SHALL provide configuration options to trade accuracy for speed (reduce map point count, matcher parameters, RANSAC iterations)

---

### Requirement 7: Diagnostic and Validation Modes

**User Story**: As a SLAM developer, I want diagnostic modes to validate PnP tracking independently, so that I can isolate issues and compare against ground truth.

#### Acceptance Criteria

1. THE System SHALL provide a `tracking_mode` configuration option with values: `'pnp'` (default), `'essential_matrix'` (legacy 2D-to-2D), `'hybrid'` (automatic fallback)
2. WHEN `tracking_mode='essential_matrix'`, THE System SHALL use only 2D-to-2D tracking (for regression testing)
3. WHEN `tracking_mode='pnp'`, THE System SHALL use only PnP tracking (fail if insufficient map points, no fallback)
4. WHEN `tracking_mode='hybrid'`, THE System SHALL use PnP with automatic fallback to 2D-to-2D
5. THE System SHALL log tracking mode transitions (PnP → fallback, fallback → PnP) with frame ID and reason
6. THE System SHALL expose metrics: PnP inlier count, reprojection error, match count, fallback frequency
7. WHERE ground truth poses are available, THE System SHALL compute per-frame pose error (translation and rotation) for validation

---

### Requirement 8: Backward Compatibility and Integration

**User Story**: As a SLAM developer, I want seamless integration with existing BA/PGO backend, so that I can deploy PnP tracking without rewriting the optimization pipeline.

#### Acceptance Criteria

1. THE PnP-estimated poses SHALL be stored in the same data structure (`self.T_world_cam`) as Essential Matrix poses
2. THE Keyframe objects SHALL contain poses in the same SE(3) format (4×4 transformation matrix)
3. THE Local Bundle Adjustment SHALL accept PnP-estimated poses as initial estimates and refine them without modification
4. THE Pose Graph Optimization SHALL accept loop closure constraints using PnP-estimated poses without modification
5. THE System SHALL preserve all existing configuration parameters for BA and PGO (no breaking changes)
6. WHEN `tracking_mode='essential_matrix'`, THE System SHALL produce identical behavior to the pre-PnP implementation (for regression testing)

---

### Requirement 9: Map Point Quality and Culling

**User Story**: As a SLAM developer, I want high-quality map points for PnP tracking, so that pose estimates are accurate and robust to outliers.

#### Acceptance Criteria

1. THE LocalMapper SHALL cull map points that fail quality checks:
   - Observed in fewer than 3 keyframes
   - Reprojection error exceeds threshold (default: 3.0px) in >50% of observations
   - Parallax angle is too small (< 1 degree) indicating poor triangulation geometry
2. THE LocalMapper SHALL prioritize map points with high observation counts for matching (more reliable)
3. WHEN a map point is observed from a new keyframe, THE LocalMapper SHALL update its descriptor (using median descriptor or best observation)
4. THE LocalMapper SHALL remove duplicate map points (3D coordinates within 0.1m and similar descriptors)
5. THE culling frequency SHALL be configurable (default: every 5 keyframes)

---

### Requirement 10: Initialization Strategy

**User Story**: As a SLAM developer, I want robust initialization that builds sufficient 3D map points before transitioning to PnP tracking, so that the system starts reliably.

#### Acceptance Criteria

1. WHEN the system starts, THE System SHALL use 2D-to-2D tracking for the first N frames (default: N=10)
2. THE initialization phase SHALL triangulate map points from the first N keyframes using Essential Matrix poses
3. THE System SHALL verify initialization success: at least 50 map points with sufficient parallax (>2 degrees)
4. WHEN initialization fails after N frames, THE System SHALL reset and retry
5. WHEN initialization succeeds, THE System SHALL log the transition: "Initialized with M map points from N keyframes, switching to PnP tracking"
6. WHERE ground truth is available, THE System SHALL optionally bootstrap initialization using GT scale to validate PnP tracking in isolation

---

## Acceptance Criteria Summary

### AC 1: 3D-to-2D Matching and PnP Estimation

**Given** a monocular SLAM system with at least 50 active 3D map points  
**When** a new frame is processed  
**Then** THE System SHALL match 2D features against 3D map points using descriptor matching  
**And** THE System SHALL estimate camera pose using `cv2.solvePnPRansac()` with at least 15 inlier correspondences  
**And** THE estimated pose SHALL be in the metric coordinate frame of the existing map  

**Validates**: Requirements 1, 2, 3

---

### AC 2: Fallback Strategy

**Given** a monocular SLAM system in PnP tracking mode  
**When** insufficient 3D map points exist (< 50) OR PnP fails to find sufficient inliers (< 15)  
**Then** THE System SHALL automatically fall back to 2D-to-2D tracking via Essential Matrix decomposition  
**And** THE System SHALL log the fallback event with reason and frame ID  
**And** THE System SHALL transition back to PnP tracking when conditions improve  

**Validates**: Requirements 4, 7

---

### AC 3: Scale Consistency During Acceleration

**Given** KITTI sequence 00 with known camera acceleration events  
**When** the camera accelerates (velocity doubles between consecutive frames)  
**Then** THE estimated depth ratio SHALL match the baseline ratio within ±20%  
**And** THE scale collapse observed in `test_scale_acceleration_bug.py` SHALL NOT occur (depth ratio ≥ 1.5 for 2× acceleration)  
**And** THE APE on KITTI sequence 00 SHALL reduce from ~197m to <20m  

**Validates**: Requirements 3, 5 (composite success criterion)

---

### AC 4: Performance Requirements

**Given** a monocular SLAM system running on reference hardware  
**When** processing KITTI sequence 00 at full frame rate  
**Then** THE 2D-to-3D matching SHALL complete in <10ms per frame  
**And** THE PnP RANSAC estimation SHALL complete in <15ms per frame  
**And** THE total tracking overhead SHALL be <30ms per frame  
**And** THE System SHALL log a warning if tracking exceeds 50ms per frame  

**Validates**: Requirement 6

---

### AC 5: Backward Compatibility

**Given** a monocular SLAM system with `tracking_mode='essential_matrix'`  
**When** running on KITTI sequence 00  
**Then** THE System SHALL produce identical behavior to the pre-PnP implementation  
**And** ALL existing unit tests for BA, PGO, and feature tracking SHALL pass without modification  
**And** THE Keyframe and MapPoint data structures SHALL remain unchanged  

**Validates**: Requirement 8

---

### AC 6: Diagnostic Modes

**Given** a monocular SLAM system with diagnostic mode enabled  
**When** `tracking_mode` is set to `'pnp'`, `'essential_matrix'`, or `'hybrid'`  
**Then** THE System SHALL enforce the specified tracking strategy  
**And** THE System SHALL log tracking mode transitions with frame ID and reason  
**And** THE System SHALL expose metrics: PnP inlier count, reprojection error, match count, fallback frequency  
**And** WHERE ground truth is available, THE System SHALL compute per-frame pose error  

**Validates**: Requirement 7

---

### AC 7: Map Point Quality

**Given** a LocalMapper with active 3D map points  
**When** processing a sequence of keyframes  
**Then** THE LocalMapper SHALL cull low-quality map points (< 3 observations, high reprojection error, poor parallax)  
**And** THE LocalMapper SHALL remove duplicate map points (< 0.1m distance, similar descriptors)  
**And** THE active map point count SHALL stabilize after initialization (no runaway growth)  

**Validates**: Requirement 9

---

### AC 8: Initialization Success

**Given** a monocular SLAM system starting from scratch  
**When** processing the first 10 frames of KITTI sequence 00  
**Then** THE System SHALL initialize using 2D-to-2D tracking for the first 10 frames  
**And** THE System SHALL triangulate at least 50 map points with sufficient parallax (>2 degrees)  
**And** THE System SHALL transition to PnP tracking after successful initialization  
**And** THE System SHALL log: "Initialized with M map points from N keyframes, switching to PnP tracking"  

**Validates**: Requirement 10

---

## Out of Scope

The following are explicitly NOT addressed by this feature:

- **Stereo or RGB-D Tracking**: This feature targets monocular SLAM only (scale from motion, not direct depth sensing)
- **Direct Image Alignment**: This feature uses feature-based tracking (not direct photometric tracking)
- **IMU Integration**: No inertial measurements are used (pure visual odometry)
- **Relocalization**: This feature does not address lost-tracking recovery (separate relocalization module required)
- **Multi-Session Mapping**: This feature operates on single-session sequences (no map merging or lifelong mapping)
- **Dynamic Object Filtering**: This feature assumes static scenes (no semantic segmentation or motion segmentation)
- **Real-Time Visualization**: Performance focus is on correctness, not rendering (separate visualization module)

---

## Dependencies and Constraints

- **Python Version**: 3.8+ (NumPy, OpenCV available)
- **OpenCV Version**: 4.5+ (for `cv2.solvePnPRansac()` stability)
- **External Libraries**: OpenCV (cv2), NumPy, SciPy
- **Hardware**: Reference hardware is modern CPU (Intel i7 or equivalent), no GPU acceleration assumed
- **Dataset**: Primary validation on KITTI Odometry dataset (sequences 00-10), ground truth poses available
- **Backward Compatibility**: Must preserve existing API for BA, PGO, feature detection, and diagnostic modes

---

## Metrics for Success

1. **Scale Consistency**: Depth ratio matches baseline ratio within ±20% during acceleration (fixes `test_scale_acceleration_bug.py`)
2. **APE Reduction**: From ~197m to <20m on KITTI sequence 00 (same target as bugfix spec)
3. **PnP Success Rate**: ≥95% of frames successfully tracked via PnP (not fallback) after initialization
4. **Inlier Count**: Median PnP inlier count ≥30 (indicates robust correspondences)
5. **Performance**: Total tracking overhead <30ms per frame (real-time capable at 30 Hz)
6. **No Regressions**: Essential Matrix mode produces identical results to pre-PnP implementation
7. **Initialization Success**: ≥98% initialization success rate on KITTI sequences (10 frames sufficient)

---

## References

- **Exploration Test**: `vo_slam/test_scale_acceleration_bug.py` — demonstrates scale collapse bug with parametrized and property-based tests
- **Bugfix Spec**: `.kiro/specs/monocular-slam-trajectory-drift/` — addresses scale estimation noise and loop detection (insufficient for root cause)
- **ORB-SLAM2 Paper**: Mur-Artal & Tardós (2017) — reference implementation uses 3D-to-2D tracking extensively
- **PnP Algorithms**: Lepetit et al. (2009) — EPnP algorithm used by OpenCV `solvePnP()`
- **KITTI Dataset**: Geiger et al. (2012) — benchmark for visual odometry and SLAM
