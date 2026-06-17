# Implementation Plan: Monocular SLAM Scale Estimation & Loop Closure Detection

## Overview

This implementation plan addresses trajectory drift in monocular SLAM through two interrelated fixes:
1. **Robust outlier-resistant depth filtering** for scale estimation (fixes Bug Condition 1.1, 1.4)
2. **Refined loop closure detection parameters** to enable more frequent corrections (fixes Bug Condition 1.2, 1.3)

The workflow follows the bug condition methodology: exploration tests confirm the bug exists, preservation tests capture non-buggy behavior, implementation applies the fix, and validation ensures both fixes work and no regressions occur.

### Workflow Summary

| Phase | Purpose | Code State | Expected Result |
|-------|---------|-----------|-----------------|
| **Task 1** | Exploration Test | UNFIXED | Test FAILS (confirms bug exists) |
| **Task 2** | Preservation Tests | UNFIXED | All tests PASS (baseline captured) |
| **Task 3** | Implementation | FIXED | Both fixes applied |
| **Task 3.4** | Verify Exploration | FIXED | Exploration test now PASSES (bug fixed) |
| **Task 3.5** | Verify Preservation | FIXED | All preservation tests still PASS (no regressions) |
| **Task 3.6** | Integration Test | FIXED | End-to-end validation passes |
| **Task 4** | Checkpoint | FIXED | All metrics verified; ready to close |

## Tasks

---

## Phase 1: Bug Exploration (Confirm Bug Exists)

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Scale Drift and Loop Closure Suppression
  - **CRITICAL**: This test MUST FAIL on unfixed code — failure confirms the bugs exist
  - **GOAL**: Surface counterexamples that demonstrate scale drift accumulation and insufficient loop closure detections
  - **Test Strategy**: Run KITTI sequence 00 (4,541 frames) with autonomous scaling enabled
    - Measure cumulative translation scale drift over 500-frame windows
    - Count loop closure detections across the full sequence
    - Track system state at frame 4000+ to confirm LOST state
  - **Bug Condition Details** (from bugfix.md 1.1-1.4):
    - Bug C(X) occurs when: depth values include outliers + temporal dead zone suppresses valid closures
    - Expected: Scale estimates stabilize within σ² ≤ 0.02, ≥ 2 distinct loop closures detected, system remains tracked through frame 4000
    - Actual: Scale drift exceeds ±5% over 500-frame windows, <2 loop closures detected, system LOST after ~4000 frames
  - **Scoped PBT**: Test with concrete KITTI 00 sequence data
    - Property: `drift_over_window(frames 0-500) ≤ 5% AND drift_over_window(frames 500-1000) ≤ 5% AND ... AND loop_closure_count ≥ 2 AND system_state != LOST at frame 4000`
    - Measure scale factors frame-by-frame via autonomous estimation
    - Count unique loop closure events triggered (revisit window detections)
    - Record trajectory at frame 4000 to confirm LOST condition
  - **Run on UNFIXED code**:
    - **EXPECTED OUTCOME**: Test FAILS with counterexamples like:
      - "Scale drift in frames 1000-1500: +8.3% (exceeds ±5% threshold)"
      - "Loop closures detected: 0 (expected ≥ 2)"
      - "System state at frame 4000: LOST (tracking failure)"
  - **Document Root Cause**:
    - Outlier-influenced median depth produces noisy scale; compounded frame-to-frame
    - min_loop_gap=200 + high detection thresholds suppress loop corrections
    - Accumulated scale error causes pose divergence, system lost
  - Mark task complete when test is written, executed on unfixed code, and failures are documented
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

---

## Phase 2: Preservation Tests (Capture Non-Buggy Behavior)

- [x] 2. Write preservation property tests
  - **Property 2: Preservation** - Valid Depth Estimation and Loop-Free Tracking
  - **CRITICAL**: Follow observation-first methodology — run unfixed code first to observe actual behavior
  - **GOAL**: Write property-based tests capturing behavior that should NOT change after the fix
  - **Observation Phase** (Run on UNFIXED code):
    - Observe: When depth values are clean (no outliers), median produces reasonable scale estimates
    - Observe: Local Bundle Adjustment and keyframe tracking work correctly without loop closures
    - Observe: Features are extracted and matched consistently across frames
    - Observe: Non-loop-closure frames process normally without suppression side effects
    - Record observed scale variance, tracking success rate, feature matching stability
  - **Preservation Requirements** (from bugfix.md 3.1-3.7):
    - 3.1: Invalid pose fallback (scale = 1.0) when broken ego-motion
    - 3.2: Diagnostic mode (scale_mode='gt') derives from ground truth unchanged
    - 3.3: Fixed/unit scale modes unmodified (scale_mode='fixed', 'none')
    - 3.4: Local BA and PGO stages operate unchanged
    - 3.5: Loop-free frame processing (features, tracking) unmodified
    - 3.6: Normal depth distributions (within parallax/bounds) produce correct scale after filtering
    - 3.7: Loop closures with sufficient temporal separation fire normally
  - **Property-Based Tests** (Recommendation: Use property-based framework like Hypothesis, QuickCheck, or fast-check):
    - **Test 1: Invalid Ego-Motion Fallback**
      - Property: For frames where PnP confidence is below threshold OR tracking fails
      - Expected: Scale estimate returns 1.0 (fallback)
      - Implementation: Generate synthetic frames with degraded features; verify fallback behavior
    - **Test 2: Diagnostic Mode Preservation**
      - Property: When scale_mode='gt', scale_factor == ground_truth_relative_baseline
      - Expected: Scaling from ground truth works correctly across all frames
      - Implementation: Run with scale_mode='gt'; verify scale matches GT for random sample of frames
    - **Test 3: Clean Depth Stability**
      - Property: For depth values within normal range (no detected outliers), median scale remains stable (σ² ≤ observed_unfixed)
      - Expected: Normal depths produce reasonable scale without filtering degrading estimates
      - Implementation: Run frames with inlier-only map points; measure scale variance matches unfixed behavior
    - **Test 4: Feature Tracking Consistency**
      - Property: Feature extraction and matching success rates unchanged on non-loop frames
      - Expected: Same features detected, same inlier percentages
      - Implementation: Compare feature counts and matching ratios across frames before/after fix
    - **Test 5: Local BA Convergence**
      - Property: Local bundle adjustment converges with same iteration counts and residual profiles
      - Expected: BA behavior identical (no regression in optimization)
      - Implementation: Log BA statistics; verify convergence metrics unchanged
  - **Run Tests on UNFIXED code**:
    - **EXPECTED OUTCOME**: All preservation tests PASS
    - Document observed baseline metrics (scale variance, tracking rates, BA convergence)
  - **After Fix**: Re-run same tests — must still PASS (confirms no regressions)
  - Mark task complete when tests are written, run on unfixed code, and all pass
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

---

## Phase 3: Implementation

- [x] 3. Fix scale estimation and loop closure detection

  - [x] 3.1 Implement robust outlier-resistant depth filtering for scale estimation
    - **Specification Reference**: Expected Behavior 2.1 & 2.4 (from bugfix.md)
    - **Task**: Replace naive median-of-depths with statistical outlier-robust filtering
    - **Implementation Details**:
      - **Method 1 (Recommended: IQR-based Tukey Fences)**:
        - Compute Q1, Q3 (25th and 75th percentile) of recent map point depths
        - IQR = Q3 - Q1
        - Lower fence: Q1 - 1.5 × IQR; Upper fence: Q3 + 1.5 × IQR
        - Reject depths outside fences; compute median on inliers only
        - Ensures 99.3% of normal distribution retained while removing outliers
      - **Method 2 (Alternative: MAD-based scoring)**:
        - Compute median and Median Absolute Deviation (MAD) of depths
        - Modified Z-score: |depth - median| / (1.4826 × MAD)
        - Reject depths with modified Z-score > 2.5
      - **Implementation Location**: Scale recovery function (compute_scale_from_depth or equivalent)
      - **Parameters**:
        - min_inliers_for_scale: Ensure at least 10 valid depths remain after filtering (fallback to 1.0 if too few)
        - outlier_rejection_method: 'iqr' (default) or 'mad'
        - iqr_multiplier: 1.5 (standard Tukey fence)
    - **Code Changes**:
      - Add outlier filtering function before median computation
      - Log inlier/outlier counts for debugging
      - Preserve fallback behavior (return 1.0 if insufficient inliers)
    - **Testing**:
      - Verify outlier filtering correctly rejects erroneous depths
      - Verify clean depth distributions pass through unmodified
      - Verify scale variance stabilizes (σ² ≤ 0.02 on KITTI 00)
    - **Verification Against Requirements**:
      - 2.1: Robust filtering applied ✓
      - 2.4: Outlier detection prevents outlier-influenced medians ✓
      - 3.1: Fallback to 1.0 when insufficient depths ✓
      - 3.6: Normal depths still produce correct estimates ✓
    - _Bug_Condition: isBugCondition = (depth_has_outliers AND no_outlier_filtering) [bugfix.md 1.1, 1.4]_
    - _Expected_Behavior: expectedBehavior = (scale_variance ≤ 0.02 AND drift ≤ ±5% over 500-frame windows) [bugfix.md 2.1, 2.4]_
    - _Preservation: maintain_fallback_scale_1_0, preserve_clean_depth_estimates [bugfix.md 3.1, 3.6]_
    - _Requirements: 1.1, 1.4, 2.1, 2.4, 3.1, 3.6_

  - [x] 3.2 Refine loop closure detection thresholds and temporal dead zone
    - **Specification Reference**: Expected Behavior 2.2 & 2.3 (from bugfix.md)
    - **Task**: Reduce temporal dead zone and refine BoW matching thresholds to enable more frequent loop closures
    - **Implementation Details**:
      - **Temporal Dead Zone Adjustment**:
        - Current: min_loop_gap = 200 frames (overly conservative)
        - New: min_loop_gap = 80 frames (or adaptive to sequence framerate)
        - Rationale: Allows more frequent corrections while preventing redundant constraint flooding
        - Adaptive formula (optional): min_loop_gap = max(80, int(framerate * 2.5)) [2.5 seconds at typical 30fps = ~75 frames]
      - **BoW Score Threshold Refinement**:
        - Current: min_bow_score = 0.012 (too high, filters valid candidates)
        - New: min_bow_score = 0.008 (relaxed to capture more candidates)
        - Preserve geometric verification downstream to reject false positives
      - **Geometric Inliers Threshold**:
        - Current: min_geo_inliers = 30 (reasonable, but review in context of relaxed BoW)
        - New: Keep at 30 or reduce to 25 if BoW relaxation produces low-quality candidates
        - Ensure sufficient inliers for robust pose estimation
      - **Dead Zone Logic**:
        - After loop closure correction: lock dead zone for min_loop_gap frames
        - Outside dead zone: loop detector runs normally (no suppression)
        - Multiple closures in same window: fire all (don't suppress valid re-detections)
    - **Code Changes**:
      - Update min_loop_gap parameter in loop detector configuration
      - Update min_bow_score in Bag-of-Words matcher
      - Modify dead zone logic to prevent suppression of valid detections
      - Add logging to track loop closure frequency and dead zone fires
    - **Testing**:
      - Verify loop closure count increases to ≥ 2 on KITTI 00
      - Verify dead zone prevents excessive redundant constraints (no 100+ constraints per window)
      - Verify false positives are still rejected by geometric verification
    - **Verification Against Requirements**:
      - 2.2: Refined parameters enable more loop closures ✓
      - 2.3: Dead zone prevents excessive firing but allows re-correction ✓
      - 3.7: Loop closures with temporal separation fire as expected ✓
    - _Bug_Condition: isBugCondition = (high_detection_thresholds AND wide_temporal_dead_zone) [bugfix.md 1.2, 1.3]_
    - _Expected_Behavior: expectedBehavior = (loop_closure_count ≥ 2 AND PGO_triggers_per_window ≤ 1) [bugfix.md 2.2, 2.3]_
    - _Preservation: prevent_redundant_constraint_flooding, maintain_pose_stability [bugfix.md 3.7]_
    - _Requirements: 1.2, 1.3, 2.2, 2.3, 3.7_

  - [x] 3.3 Integrate outlier filtering and loop closure improvements into full pipeline
    - **Task**: Combine fixes from 3.1 and 3.2; test integration
    - **Implementation Details**:
      - Ensure robust depth filtering runs in scale estimation path
      - Ensure refined loop closure parameters are loaded at initialization
      - Add configuration validation to reject invalid parameter combinations
      - Enable diagnostic logging for scale drift and loop closure events
    - **Testing**:
      - End-to-end test on KITTI 00 with both fixes enabled
      - Verify scale and loop closure improvements compound (drift + closure count)
      - Verify no conflicts between scale estimation and loop closure correction
    - **Code Changes**:
      - Config file updates (if needed)
      - Integration test
      - Logging/diagnostics
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4_

---

## Phase 4: Validation

- [x] 3.4 Verify bug condition exploration test now passes
  - **Property 1: Expected Behavior** - Scale Stability and Loop Closure Success
  - **CRITICAL**: Re-run the SAME test from task 1 — do NOT write a new test
  - **IMPORTANT**: The test from task 1 encodes the expected behavior; passing confirms the fix works
  - **Process**:
    - Re-run full KITTI 00 sequence with autonomous scaling and refined loop detection
    - Measure scale drift over 500-frame windows: verify all windows ≤ ±5%
    - Count loop closures: verify ≥ 2 distinct detections
    - Track system state at frame 4000: verify NOT LOST
    - Verify APE/RPE metrics improve (drift < 100m APE target)
  - **Expected Outcome**: Test PASSES with results like:
    - "All 500-frame windows: drift ≤ ±4.8% ✓"
    - "Loop closures detected: 2+ ✓"
    - "System state at frame 4000: TRACKED ✓"
    - "APE: < 100m (improved from 50-350m baseline) ✓"
  - **Acceptance Criteria**:
    - Scale drift: ALL 500-frame windows ≤ ±5%
    - Loop closures: ≥ 2 distinct detections
    - System state: Remains TRACKED through frame 4000
    - APE/RPE: Significantly improved vs baseline
  - Mark task complete when test passes and metrics meet criteria
  - _Requirements: 2.1, 2.2, 2.4_

- [x] 3.5 Verify preservation tests still pass
  - **Property 2: Preservation** - Unchanged Behavior Verified
  - **CRITICAL**: Re-run the SAME tests from task 2 — do NOT write new tests
  - **Process**:
    - Re-run all preservation tests from task 2 on fixed code
    - Verify all tests still pass (no regressions)
    - Compare metrics to unfixed baseline: should match within tolerance
  - **Test Verification**:
    - Invalid ego-motion fallback: Still returns scale 1.0 ✓
    - Diagnostic mode (scale_mode='gt'): Still matches ground truth ✓
    - Clean depth stability: Still produces reasonable estimates ✓
    - Feature tracking: Extraction/matching rates unchanged ✓
    - Local BA: Convergence metrics unchanged ✓
  - **Acceptance Criteria**:
    - All 5 preservation tests PASS
    - No metric regression (variance, tracking rates, BA residuals unchanged)
  - mark task complete when all preservation tests pass
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 3.6 Comprehensive integration test: Scale + Loop Closure Synergy
  - **Task**: Verify fixes work together holistically
  - **Process**:
    - Run full KITTI sequence with both robust depth filtering and refined loop closure
    - Confirm scale estimates stabilize (outliers removed)
    - Confirm loop closures fire frequently enough to correct drift
    - Confirm PGO (Pose Graph Optimization) applies corrections smoothly
    - Measure final APE/RPE: target < 100m APE (vs 50-350m baseline)
    - Verify no optimizer divergence or self-folding
  - **Success Criteria**:
    - APE < 100m ✓
    - Smooth trajectory correction (no discontinuities from PGO) ✓
    - All tests pass (exploration + preservation) ✓
  - _Requirements: 1.1-1.4, 2.1-2.4, 3.1-3.7_

---

## Phase 5: Checkpoint

- [x] 4. Checkpoint - Ensure all tests pass
  - **Task**: Final verification before marking bugfix complete
  - **Checklist**:
    - [x] Bug condition exploration test: PASSES (confirms bug fixed)
    - [x] All preservation tests: PASS (confirms no regressions)
    - [x] Integration test: PASSES (confirms synergy)
    - [x] APE/RPE metrics: Improved vs baseline (< 100m APE target met)
    - [x] Code compiles without warnings
    - [x] All diagnostics logs clean (no unexpected errors)
  - **Acceptance**: All checkboxes above must be satisfied
  - **Resolution**: If any item fails:
    - [x] Rerun task 3.x to investigate and fix issue
    - [x] Document root cause
    - [x] Re-validate with tests
  - mark task complete when all items verified

---

## Testing Infrastructure Notes

### Property-Based Testing Framework Recommendation
- **For C++**: Use a C++ property-based testing framework (e.g., RapidCheck, or custom harness)
- **For Python (if used for utilities)**: Use Hypothesis
- **Approach**: Generate synthetic test cases that exercise the bug condition and preservation requirements

### Test Datasets
- **Primary**: KITTI Sequence 00 (4,541 frames, known loop closure at end)
- **Validation**: Consider KITTI sequences 01-10 for robustness (optional, if time permits)

### Metrics to Track
- **Scale**: Drift % per 500-frame window, variance σ²
- **Loop Closures**: Count, temporal spacing, detector confidence
- **Trajectory**: APE (Absolute Pose Error), RPE (Relative Pose Error)
- **System State**: TRACKED vs LOST at key frame indices

---

## Task Dependency Graph

```json
{
  "waves": [
    {
      "id": 1,
      "name": "Bug Confirmation (Unfixed Code)",
      "tasks": ["1", "2"]
    },
    {
      "id": 2,
      "name": "Implementation & Validation (Fixed Code)",
      "tasks": ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6"]
    },
    {
      "id": 3,
      "name": "Final Checkpoint",
      "tasks": ["4"]
    }
  ],
  "taskDetails": [
    {
      "id": "1",
      "name": "Exploration Test (Bug Condition)",
      "runsOn": "UNFIXED",
      "expectedResult": "FAIL (confirms bug exists)"
    },
    {
      "id": "2",
      "name": "Preservation Tests",
      "runsOn": "UNFIXED",
      "expectedResult": "PASS (baseline captured)"
    },
    {
      "id": "3.1",
      "name": "Robust Depth Filtering",
      "runsOn": "FIXED",
      "expectedResult": "Complete"
    },
    {
      "id": "3.2",
      "name": "Loop Closure Thresholds",
      "runsOn": "FIXED",
      "expectedResult": "Complete"
    },
    {
      "id": "3.3",
      "name": "Pipeline Integration",
      "runsOn": "FIXED",
      "expectedResult": "Complete"
    },
    {
      "id": "3.4",
      "name": "Verify Exploration Test",
      "runsOn": "FIXED",
      "expectedResult": "PASS (bug fixed)"
    },
    {
      "id": "3.5",
      "name": "Verify Preservation Tests",
      "runsOn": "FIXED",
      "expectedResult": "PASS (no regressions)"
    },
    {
      "id": "3.6",
      "name": "Integration Test",
      "runsOn": "FIXED",
      "expectedResult": "PASS (metrics met)"
    },
    {
      "id": "4",
      "name": "Checkpoint",
      "runsOn": "FIXED",
      "expectedResult": "All checks verified"
    }
  ]
}
```

## Notes

### Key Workflow Principles

1. **Exploration Before Fix** (Task 1):
   - Write property-based test encoding the bug condition
   - Run on UNFIXED code — test MUST FAIL (confirms bug exists)
   - Document counterexamples to understand root cause
   - Do NOT fix the test or code during this phase

2. **Preservation Baseline** (Task 2):
   - Follow observation-first methodology: run unfixed code, observe behavior
   - Write property-based tests capturing observed behavior on non-buggy inputs
   - Verify tests PASS on unfixed code (establishes baseline)
   - After fix: re-run same tests — must still PASS (no regressions)

3. **Implementation** (Task 3):
   - Apply both fixes (depth filtering + loop closure thresholds)
   - Code changes are isolated to scale estimation and loop detector modules
   - Integrate both fixes in a single coherent implementation

4. **Validation** (Tasks 3.4, 3.5, 3.6):
   - Re-run exploration test from Task 1 → must now PASS (bug fixed)
   - Re-run preservation tests from Task 2 → must still PASS (no regressions)
   - Run comprehensive integration test → confirm synergy

5. **Final Checkpoint** (Task 4):
   - All tests passing
   - Metrics meet targets (APE < 100m, scale variance ≤ 0.02, ≥ 2 loop closures)
   - No regressions in non-buggy behavior

### Expected Metrics After Fix

| Metric | Current (Buggy) | Target (Fixed) | Source |
|--------|-----------------|----------------|--------|
| Scale Drift (500-frame window) | ±8-15% | ≤ ±5% | bugfix.md 2.1 |
| Scale Variance (σ²) | > 0.05 | ≤ 0.02 | bugfix.md 2.1 |
| Loop Closures (KITTI 00) | < 2 | ≥ 2 | bugfix.md 2.2 |
| APE (Absolute Pose Error) | 50-350m | < 100m | bugfix.md Introduction |
| System State at Frame 4000 | LOST | TRACKED | User feedback |
| Temporal Dead Zone | 200 frames | 80 frames | bugfix.md 2.2 |

### Implementation Guidance

**Depth Filtering (Task 3.1)**: Use IQR-based Tukey fences as the primary approach:
- Compute Q1, Q3 of map point depths
- IQR = Q3 - Q1
- Reject depths outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- Compute median on remaining inliers
- Fallback to scale=1.0 if < 10 inliers remain

**Loop Closure Thresholds (Task 3.2)**:
- Reduce min_loop_gap from 200 → 80 frames
- Reduce min_bow_score from 0.012 → 0.008
- Keep min_geo_inliers at 30 (or reduce to 25 if needed)
- Modify dead zone logic to prevent suppression of valid re-detections outside the window

### Testing Infrastructure

- **Framework**: Use property-based testing (C++ RapidCheck, or custom harness)
- **Dataset**: KITTI Sequence 00 (primary test case with known loop closure)
- **Logging**: Enable diagnostics to track scale, loop closures, and system state
- **Metrics**: Monitor APE, RPE, scale variance, loop closure count, system state

### Files Modified

- **Scale Estimation Module**: Add outlier-resistant depth filtering
- **Loop Detector Module**: Update thresholds and dead zone logic
- **Configuration (if separate)**: May need to add/update parameters
- **Diagnostics/Logging**: Add tracking for scale and loop closure events

### Rollback Plan

If issues arise during implementation:
1. Revert to last known good state
2. Identify root cause through diagnostic logs
3. Revise implementation strategy
4. Re-test with both exploration and preservation tests

---

## References

- **Bug Specification**: bugfix.md (Requirements 1.x for Bug Condition, 2.x for Expected Behavior, 3.x for Preservation)
- **KITTI Dataset**: `/Users/anirudhsinghair/Documents/GitHub/VisualSLAM/KITTI/sequences/00/`
- **Depth Filtering Techniques**:
  - Tukey Fences (IQR-based): Practical & robust for outlier removal
  - MAD (Median Absolute Deviation): Robust alternative to standard deviation
  - Z-score Filtering: Common for Gaussian distributions
- **Loop Closure Detection**: Bag-of-Words (BoW) matching + geometric verification (PnP)
