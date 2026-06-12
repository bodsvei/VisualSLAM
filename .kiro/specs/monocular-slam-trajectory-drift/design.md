# Monocular SLAM Trajectory Drift Bugfix Design

## Overview

This design formalizes the approach to fix monocular SLAM trajectory drift on KITTI sequence 00, reducing APE from ~197m to <20m. The bug manifests as continuous error accumulation throughout long sequences due to two root causes:

1. **Scale Estimation Noise**: Median depth filtering lacks outlier rejection, allowing spurious depth values to corrupt scale recovery
2. **Insufficient Loop Closure Constraints**: Conservative loop detector parameters miss valid loop closures that would provide global corrections

The fix employs a two-pronged strategy:
- **Robust Depth Filtering**: Use IQR-based outlier detection (Tukey fences) to reject outliers before median computation
- **Strengthened Loop Detection**: Reduce temporal dead zone (200→100 frames) and BoW threshold (0.012→0.009) to catch more valid loops

Both fixes are minimal, targeted changes that preserve diagnostic modes, fallback behavior, and compatibility with Local BA and PGO.

---

## Glossary

- **Bug_Condition (C)**: A set of inputs that trigger the bug—depths containing outliers in scale recovery, OR loop closure opportunities missed due to conservative parameters
- **Property (P)**: The desired correct behavior when the bug condition holds—robust scale estimates via outlier-filtered median, AND increased loop closure verification
- **Preservation**: Existing behavior that must NOT change—diagnostic modes, feature tracking, edge case handling, fallback to 1.0
- **Inlier**: A depth value within the central distribution (not filtered as outlier)
- **Outlier**: A depth value far from the central distribution (typically >1.5×IQR beyond Q1 or Q3)
- **IQR (Interquartile Range)**: Q3 - Q1, where Q1 and Q3 are 25th and 75th percentiles
- **Tukey Fences**: Standard outlier bounds using IQR: lower = Q1 - 1.5×IQR, upper = Q3 + 1.5×IQR
- **BoW (Bag of Words)**: Visual vocabulary-based place recognition scoring
- **Temporal Dead Zone**: Suppression period after verified loop to prevent callback flooding
- **Geometric Verification**: PnP RANSAC with inlier count threshold to confirm loops

---

## Bug Details

### Bug Condition

The bug manifests when the monocular SLAM system processes long sequences with the following conditions:

**Condition 1 - Scale Estimation Bug:**
```
FUNCTION isBugCondition_ScaleEstimation(input)
  INPUT: input is a sequence of frames with map_points
  OUTPUT: boolean
  
  RETURN scale_mode == 'median_depth'
         AND recent_map_points.size() > 0
         AND existsOutlier(depths(recent_map_points, T_cam_world))
         AND median(depths_with_outliers) produces_incorrect_scale
END FUNCTION
```

**Condition 2 - Loop Detection Bug:**
```
FUNCTION isBugCondition_LoopDetection(input)
  INPUT: input is a sequence with loop closure opportunities
  OUTPUT: boolean
  
  RETURN exists_valid_loop_closure
         AND frames_since_last_verification < min_loop_gap_frames (200)
         OR bow_score > min_bow_score_threshold (0.012)
         AND loop_is_not_detected
END FUNCTION
```

**Concrete Manifestation:**

In KITTI sequence 00:
- Scale recovery encounters outlier depths from triangulation failures (e.g., points in untextured regions, matching errors)
- Example: Recent depths = [2.8m, 3.0m, 2.9m, 150.0m, 3.1m] — the 150m outlier (triangulation failure) pulls median toward extreme value
- Incorrect scale corrupts camera poses → accumulated drift grows continuously
- Simultaneously, loop detector misses valid revisits due to conservative parameters (200-frame dead zone, 0.012 BoW threshold)
- Result: Drift is never corrected by global optimization

### Examples

**Example 1: Scale Estimation Outlier Scenario**

*Frame 500, `scale_mode='median_depth'`*
- Recent 200 map points sampled
- Depths: [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 150.0] (9 inliers, 1 extreme outlier)
- Current behavior: median([...150.0]) ≈ 5.0 (skewed by outlier)
- Fixed behavior: IQR filter removes 150.0 → median([2.5...3.3]) ≈ 2.9 (robust)
- Impact: Scale factor changes from 5.0 to 2.9, significantly correcting pose estimate

**Example 2: Loop Detection Miss**

*Frame 2000 (revisiting frame 1900, 100 frames earlier)*
- Last verified loop at frame 1850
- Time since last loop: 150 frames
- Current logic: Suppress because 150 < 200 (min_loop_gap_frames)
- Fixed logic: Allow because min_loop_gap_frames reduced to 100 → detect loop, trigger PGO correction

**Example 3: Edge Case — Few Map Points**

*Frame 50 (early in sequence, few triangulated points)*
- Recent map points: only 3 points
- Current: median([d1, d2, d3]) — noisy estimate
- Fixed: Check count < 5 → return 1.0 (defer to next frame when more points available)

**Example 4: Edge Case — All Outliers**

*Pathological frame with poor triangulation*
- Recent depths: [100.0, 200.0, 300.0] (all beyond physical range)
- IQR analysis: Q1=150, Q3=250, IQR=100, lower=-0, upper=400
- After filter: all points outside range due to geometric/measurement anomaly
- Result: Return 1.0 (no scale correction, wait for next frame)

---

## Expected Behavior

### Preservation Requirements

The fix must NOT change the following behaviors:

**Unchanged Behaviors:**
1. **Diagnostic Modes**: When `scale_mode='gt'`, `'fixed'`, or `'none'`, scale recovery produces identical results
2. **Feature Tracking**: Feature detection, description, matching counts unchanged
3. **Single-Point Scenario**: When `map_points` is empty, always return 1.0
4. **Non-Finite Pose**: When `T_cam_world` contains NaN/inf, always return 1.0
5. **Local Bundle Adjustment**: BA convergence behavior unaffected by scale changes
6. **Edge Case Robustness**: Graceful handling of pathological inputs (no crashes)

**Scope:**
All inputs that do NOT involve:
- Scale recovery with outlier-contaminated depths, OR
- Loop detection with conservatively suppressed candidates

...should be completely unaffected. This includes:
- Feature extraction and tracking
- Initial pose estimation
- Keyframe selection
- Map point culling
- BA and PGO algorithms (only inputs change, algorithms unchanged)

---

## Hypothesized Root Cause

Based on the bug description and code analysis, the most likely root causes are:

### 1. **Outlier Depths in Scale Recovery**

**Analysis**: 
- `_recover_scale()` computes median of 200 recent map points without filtering
- Triangulation can produce spurious depth estimates due to:
  - Feature matches in near-identical appearance regions (ambiguous triangulation)
  - Failures to track features across baseline (incorrect correspondence)
  - Numerical issues from DLT solver in degenerate configurations
  - Points near epipolar plane (high reprojection error but not filtered due to relax clamping)

**Evidence**:
- Current code: `depth = compute_median_depth(recent, T_cur_world)` followed by clamp to [0.3, 80.0]m
- Clamping alone is insufficient—an outlier at 150m still influences median before clamp
- Example: median([2.8, 2.9, 3.0, 3.1, 150]) ≈ 5.0 → clamp to 80.0 → still wrong

### 2. **Conservative Loop Detector Parameters**

**Analysis**:
- `min_loop_gap_frames=200` (6.67s at 30 Hz) is excessive for KITTI sequences (~130s total)
- Many valid revisits occur within 200 frames (e.g., circling a parking lot, returning after detour)
- `min_bow_score=0.012` requires very high descriptor similarity, rejecting marginal matches

**Evidence**:
- KITTI sequence 00: vehicle makes multiple loops, but temporal dead zone suppresses detections
- Example: If loop verified at frame 500, next loop suppressed until frame 700
  - But vehicle may revisit frame 600 (valid loop) and frame 800 (different loop)
  - Current: Frame 600 revisit suppressed (within 200 frames)
  - Fixed: Frame 600 revisit detected (within 100 frames), frame 800 also detectable

### 3. **Insufficient Global Constraints**

**Analysis**:
- Without loop closures, SLAM accumulates only local (incremental) constraints
- Each pose estimate compounds error from previous frame
- Over 4541 frames, even small per-frame errors (1cm) accumulate to >100m

**Evidence**:
- With few detected loops: drift uncorrected
- With more detected loops: PGO provides global constraints, redistributing error more evenly

---

## Correctness Properties

Property 1: Bug Condition - Outlier-Robust Scale Estimation

_For any_ input where outliers exist in recent map point depths (isBugCondition_ScaleEstimation returns true), the fixed `_recover_scale()` function SHALL filter depths using IQR-based outlier detection (Tukey fences, multiplier 1.5) before computing median, producing a robust depth estimate resistant to spurious values.

**Validates: Requirements AC 2.1**

Property 2: Bug Condition - Enhanced Loop Detection

_For any_ input where a valid loop closure opportunity exists within the previous 100 frames (isBugCondition_LoopDetection returns true after parameter adjustment), the fixed `LoopDetector` with `min_loop_gap_frames=100` and `min_bow_score=0.009` SHALL detect and geometrically verify the loop, triggering PGO correction.

**Validates: Requirements AC 2.2**

Property 3: Preservation - Diagnostic Modes Unchanged

_For any_ input where `scale_mode` is set to `'gt'`, `'fixed'`, or `'none'` (isBugCondition_ScaleEstimation returns false), the fixed `_recover_scale()` function SHALL produce exactly the same behavior as the original function, preserving all diagnostic capabilities.

**Validates: AC 2.3**

Property 4: Preservation - Edge Case Handling

_For any_ edge case input where map points are few (<5), all filtered as outliers, or depths are non-finite, the fixed `_recover_scale()` function SHALL return 1.0, matching the original fallback behavior and preventing crashes.

**Validates: AC 2.1**

Property 5: Preservation - Integration Compatibility

_For any_ input to the full SLAM pipeline (Local BA, PGO, feature tracking), the fixed components SHALL maintain compatibility: Local BA converges without numerical issues, PGO optimizes with increased loop frequency, and feature tracking statistics remain unchanged.

**Validates: AC 2.5**

---

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

#### File 1: `vo_slam/pipeline.py`, Function: `_recover_scale()`

**Specific Changes**:

1. **IQR-Based Outlier Detection Logic (new)**
   - After extracting recent 200 map points, compute their depths
   - Calculate Q1 (25th percentile) and Q3 (75th percentile) using NumPy
   - Compute IQR = Q3 - Q1
   - Define Tukey bounds: lower = Q1 - 1.5×IQR, upper = Q3 + 1.5×IQR
   - Filter depths to retain only those in [lower, upper]

2. **Edge Case Handling (new)**
   - If fewer than 5 map points: return 1.0 (insufficient data)
   - If all depths filtered as outliers: return 1.0 (no inliers remain)
   - If any depth is NaN or inf: filter out before IQR computation
   - If any depth is negative: filter out (impossible in camera frame)

3. **Median Computation (modified)**
   - Compute median ONLY on filtered depths (not raw)
   - If filtered list is empty, return 1.0

4. **Clamping (unchanged)**
   - After median, apply existing clamp to [scale_clamp_min, scale_clamp_max]
   - Preserve as secondary safety check

5. **Code Preservation (critical)**
   - Diagnostic modes (`'gt'`, `'fixed'`, `'none'`) unchanged—test each in unit tests
   - Fallback to 1.0 on non-finite pose: unchanged
   - Return type and range (>0, finite): unchanged

**Pseudocode**:
```
FUNCTION _recover_scale_fixed(pose, map_points, T_world_cam)
  IF scale_mode != 'median_depth' THEN
    RETURN _recover_scale_original(pose, map_points, T_world_cam)
  END IF
  
  IF map_points.length < 5 THEN
    RETURN 1.0
  END IF
  
  # Extract and filter depths
  depths := compute_depths(map_points, T_cam_world)
  depths_finite := filter(depths, isFinite AND > 0)
  
  IF depths_finite.length < 5 THEN
    RETURN 1.0
  END IF
  
  # IQR-based outlier detection
  q1 := percentile(depths_finite, 25)
  q3 := percentile(depths_finite, 75)
  iqr := q3 - q1
  lower := q1 - 1.5 * iqr
  upper := q3 + 1.5 * iqr
  
  # Filter outliers
  inliers := filter(depths_finite, d >= lower AND d <= upper)
  
  IF inliers.length < 5 THEN
    RETURN 1.0
  END IF
  
  # Compute robust median
  depth_robust := median(inliers)
  
  # Apply clamping
  scale := clamp(depth_robust, scale_clamp_min, scale_clamp_max)
  
  RETURN scale
END FUNCTION
```

#### File 2: `vo_slam/loop_detector.py`, Function: `__init__()` and class variables

**Specific Changes**:

1. **Update Default Parameters in `__init__()`**
   - Change default `min_loop_gap_frames` from 200 to **100**
   - Change default `min_bow_score` from 0.012 to **0.009**
   - Keep all other parameters unchanged

2. **Update Class Documentation (non-functional)**
   - Add comment: "FIX 4: Reduced temporal dead zone from 200 to 100 frames to detect more revisits"
   - Add comment: "FIX 5: Lowered BoW threshold from 0.012 to 0.009 for marginal matches"

3. **Code Preservation (critical)**
   - Geometric verification logic unchanged (PnP RANSAC, inlier count)
   - Callback exception handling preserved
   - Match-KF deduplication (FIX 2) preserved
   - Temporal dead zone mechanism (FIX 1) preserved — only parameter changes

**Pseudocode**:
```
CLASS LoopDetector:
  FUNCTION __init__(..., min_loop_gap_frames: int = 100, min_bow_score: float = 0.009, ...):
    # Store parameters
    self.min_loop_gap_frames := min_loop_gap_frames  # Changed default 200 → 100
    self.min_bow_score := min_bow_score            # Changed default 0.012 → 0.009
    # ... rest unchanged
  END FUNCTION
END CLASS
```

#### File 3: `vo_slam/pipeline.py`, Class: `VOConfig`

**Specific Changes (optional, for tunability)**:

1. **Add configuration parameters to VOConfig**
   - `outlier_rejection_enabled: bool = True` (enable/disable IQR filtering)
   - `iqr_multiplier: float = 1.5` (Tukey fence multiplier)
   - `min_depths_for_robust_estimate: int = 5` (minimum inliers required)
   - `loop_detector_min_gap_frames: int = 100`
   - `loop_detector_min_bow_score: float = 0.009`

2. **Pass config to LoopDetector on initialization**
   - In `VisualOdometry.__init__()`, pass `cfg.loop_detector_min_gap_frames` and `cfg.loop_detector_min_bow_score` to LoopDetector

3. **Use config in `_recover_scale()`**
   - Check `cfg.outlier_rejection_enabled` before filtering
   - Use `cfg.iqr_multiplier` and `cfg.min_depths_for_robust_estimate` in logic

**Rationale**: Allows experiments without code changes; optional but recommended

---

## Testing Strategy

### Validation Approach

The testing strategy employs a **two-phase validation cycle**:

1. **Phase A: Exploratory Bug Condition Checking** — Surface counterexamples demonstrating the bug on UNFIXED code, confirming root cause hypotheses
2. **Phase B: Fix & Preservation Checking** — Verify the fix resolves buggy inputs AND preserves non-buggy behavior

### Phase A: Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis.

**Approach**: Create unit tests that exercise the bug condition with known inputs, run on unfixed code to observe failures.

**Test Cases - Scale Estimation Bug**:

1. **Test A1: Outlier in Depth List**
   - Input: Recent depths = [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 150.0]
   - Current behavior (unfixed): median ≈ 5.0 (biased by outlier)
   - Expected (fixed): median ≈ 2.9 (outlier removed)
   - Assertion (unfixed): Scale is incorrect
   - Assertion (fixed): Scale is robust

2. **Test A2: Multiple Outliers**
   - Input: Recent depths = [2.5, 200.0, 2.6, 300.0, 2.7, 400.0, 2.8, 2.9, 3.0, 3.1]
   - Current behavior (unfixed): median heavily skewed
   - Expected (fixed): Outliers removed, median ≈ 2.9

3. **Test A3: Extreme Outlier (Failed Triangulation)**
   - Input: Recent depths = [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 1e6]
   - Current behavior (unfixed): median ≈ 5.0 (skewed)
   - Expected (fixed): Outlier filtered, median ≈ 2.9

**Test Cases - Loop Detection Bug**:

1. **Test B1: Loop Within Reduced Dead Zone**
   - Setup: Last verified loop at frame 500
   - Input: Revisit at frame 600 (100 frames later)
   - Current behavior (unfixed, min_loop_gap_frames=200): Suppressed, no event fired
   - Expected (fixed, min_loop_gap_frames=100): Loop detected, event fired

2. **Test B2: Marginal BoW Score**
   - Setup: BoW candidate with score 0.0105
   - Current behavior (unfixed, min_bow_score=0.012): Rejected
   - Expected (fixed, min_bow_score=0.009): Accepted for geometric verification

3. **Test B3: Valid Geometric Loop at Marginal BoW Score**
   - Setup: Revisit with BoW score 0.0105, PnP RANSAC produces 20 inliers
   - Current behavior (unfixed): BoW rejection → no loop event
   - Expected (fixed): BoW acceptance → geometric verification → loop event fired

**Expected Counterexamples Before Fix**:
- Scale estimates biased by outliers → large scale errors → pose drift accumulates
- Loop closures suppressed or rejected → no global optimization → drift uncorrected
- Over 4541 frames: drift compounds to ~197m

### Phase B: Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode**:
```
FOR ALL input WHERE isBugCondition(input) DO
  result_fixed := fixedFunction(input)
  ASSERT expectedBehavior(result_fixed)
END FOR
```

**Test Plan - Scale Estimation**:

After implementing IQR filtering, test:
1. Outlier-contaminated depths produce robust median
2. Clamping applied correctly after filtering
3. Multiple outliers handled correctly
4. Edge case: few map points returns 1.0
5. Edge case: all outliers returns 1.0
6. Full sequence (KITTI 00): APE reduces from ~197m toward <20m

**Test Plan - Loop Detection**:

After parameter adjustment, test:
1. Reduced dead zone increases loop detections
2. Lowered BoW threshold accepts marginal candidates
3. Geometric verification prevents false positives
4. Full sequence (KITTI 00): More loop events fired, APE reduces

**Metric: APE Trajectory Error**
- Baseline: ~197m on KITTI 00
- Target: <20m with both fixes
- Success criterion: APE reduction ≥177m

### Phase C: Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode**:
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT originalFunction(input) = fixedFunction(input)
END FOR
```

**Testing Approach**: Property-based testing and unit tests.

**Test Plan - Scale Estimation Preservation**:

1. **Diagnostic Modes**
   - `scale_mode='gt'`: Verify identical ground truth scale recovery
   - `scale_mode='fixed'`: Verify identical fixed scale return
   - `scale_mode='none'`: Verify both return 1.0

2. **Edge Cases (Preservation)**
   - Empty map_points: Both return 1.0
   - Non-finite pose: Both return 1.0
   - Single map point: Both return 1.0

3. **Feature Tracking**
   - Feature detection counts unchanged
   - Descriptor matching unchanged
   - Tracking success rates unchanged

**Test Cases - Scale Estimation Preservation**:

1. **Test C1: Diagnostic Mode - GT Scale**
   ```
   Input: scale_mode='gt'
   Expected: originalFunction(scale_mode='gt') = fixedFunction(scale_mode='gt')
   ```

2. **Test C2: Diagnostic Mode - Fixed Scale**
   ```
   Input: scale_mode='fixed', fixed_scale=2.5
   Expected: Both return 2.5
   ```

3. **Test C3: Diagnostic Mode - None**
   ```
   Input: scale_mode='none'
   Expected: Both return 1.0
   ```

4. **Test C4: Clean Depths (No Outliers)**
   ```
   Input: Recent depths = [2.8, 2.9, 3.0, 3.1, 3.2] (all inliers, no outliers)
   Expected: Filtered median ≈ original median (same result)
   ```

5. **Test C5: Empty Map Points**
   ```
   Input: map_points = []
   Expected: Both return 1.0
   ```

**Test Plan - Loop Detection Preservation**:

1. **Scenes Without Loops**
   - Input: Sequence with no revisits
   - Expected: Loop detection behavior unchanged (no loops detected regardless)

2. **Feature Tracking During Loops**
   - Input: Loop closure scenario
   - Expected: Feature tracking unaffected by parameter changes

3. **Callback Exception Handling**
   - Expected: Callback exceptions still handled gracefully (thread continues)

4. **Match-KF Deduplication**
   - Expected: Same landmark still cannot trigger multiple events

### Unit Tests

**Scale Estimation Tests**:
- `test_iqr_filters_outliers()`: Verify Tukey fence correctly removes extreme values
- `test_median_robust_vs_raw()`: Compare filtered median vs. raw median on same input
- `test_edge_case_few_points()`: Verify returns 1.0 for <5 points
- `test_edge_case_all_outliers()`: Verify returns 1.0 when all filtered out
- `test_edge_case_non_finite()`: Verify NaN/inf filtered without crash
- `test_diagnostic_modes_unchanged()`: Verify 'gt', 'fixed', 'none' produce same results
- `test_clamping_applied()`: Verify [0.3, 80.0]m clamp enforced
- `test_kitti_00_ape_reduction()`: Full sequence test, APE from ~197m to <20m

**Loop Detection Tests**:
- `test_reduced_dead_zone()`: Verify 100-frame window allows more loops
- `test_lowered_bow_threshold()`: Verify 0.009 threshold accepts marginal candidates
- `test_geometric_verification()`: Verify PnP RANSAC still enforces inlier count
- `test_no_false_positives()`: Verify geometric verification prevents invalid loops
- `test_loop_event_count()`: Verify increased events compared to baseline
- `test_callback_exception_handling()`: Verify thread robustness
- `test_match_kf_deduplication()`: Verify same landmark doesn't fire multiple events

### Property-Based Tests

1. **Property: Outlier Rejection Reduces Variance**
   - Generate random depth lists with varying outlier proportions
   - Assert: filtered_depths variance ≤ raw_depths variance
   - Rationale: Outlier filtering should reduce noise

2. **Property: Median Convergence**
   - Generate large depth samples (n=1000) from normal distribution with outliers
   - Assert: filtered_median converges to true distribution mean
   - Rationale: Robust median should estimate underlying distribution

3. **Property: Loop Detection Monotonic**
   - For fixed scene, loop_event_count(min_gap=100, min_bow=0.009) ≥ loop_event_count(min_gap=200, min_bow=0.012)
   - Rationale: Relaxing parameters increases detection

4. **Property: Preservation Under Non-Buggy Inputs**
   - Generate inputs without outliers
   - Assert: filtered_median ≈ raw_median (same for clean data)
   - Rationale: Filtering should not affect inlier-only sets

### Integration Tests

1. **Test: Full KITTI 00 Sequence**
   - Setup: Run complete SLAM pipeline on KITTI sequence 00
   - Verify:
     - APE reduces from ~197m to <20m
     - Local BA converges without numerical issues
     - PGO optimizes with increased loop frequency
     - Feature tracking stats unchanged
   - Metrics: APE, loop count, BA residuals, tracking success rate

2. **Test: Loop Closure Effects on Pose Graph**
   - Setup: Verify PGO optimization with increased loop events
   - Verify: Solver converges, residuals reasonable, poses corrected

3. **Test: Diagnostic Mode Integration**
   - Setup: Run with scale_mode='gt' to isolate scale recovery effects
   - Verify: Trajectory matches ground truth (no scale ambiguity artifacts)

---

## Testing Summary

| Phase | Test Type | Input | Expected | Validates |
|-------|-----------|-------|----------|-----------|
| A | Unit | Outlier depths | Bias in unfixed code | Root cause confirmed |
| A | Unit | Suppressed loop | No event in unfixed code | Root cause confirmed |
| B | Unit | Outlier depths (fixed) | Robust median | Property 1 |
| B | Unit | Reduced dead zone (fixed) | More loops detected | Property 2 |
| B | Integration | KITTI 00 (fixed) | APE < 20m | AC 2.4 |
| C | Unit | Diagnostic modes | Identical to original | Property 3 |
| C | Unit | Edge cases | Safe fallback (1.0) | Property 4 |
| C | PBT | Non-buggy inputs | Variance unchanged | Preservation |
| C | Integration | Full pipeline | BA/PGO/tracking unchanged | Property 5 |

