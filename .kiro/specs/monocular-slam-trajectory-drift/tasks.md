# Monocular SLAM Trajectory Drift Bugfix - Implementation Tasks

## Overview

This task list implements the bugfix for monocular SLAM trajectory drift. Tasks follow the exploratory bugfix workflow:
1. **Exploration tests** (BEFORE FIX) — demonstrate bug on unfixed code
2. **Preservation tests** (BEFORE FIX) — capture baseline behavior
3. **Implementation** — apply fixes to pipeline.py and loop_detector.py
4. **Verification** (AFTER FIX) — re-run exploration and preservation tests, full validation

## Implementation Plan

- [x] 1. Write bug condition exploration test - Scale estimation
  - **Property 1: Bug Condition** - Outlier-Corrupted Depth Estimation
  - **CRITICAL**: This test MUST FAIL on unfixed code (confirms bug exists)
  - Test demonstrates outlier bias: raw median ≈ 5.0 vs. filtered median ≈ 2.9
  - Generate depth lists with outliers (e.g., [2.5-3.3, 150.0])
  - Property-based approach: Varying outlier proportions and magnitudes
  - Document counterexamples found during execution
  - Expected outcome: Test FAILS on unfixed code (this confirms bug)
  - _Requirements: 1.1, 1.2, 2.1_

- [ ] 2. Write bug condition exploration test - Loop detection
  - **Property 1: Bug Condition** - Loop Closures Suppressed by Conservative Parameters
  - **CRITICAL**: Demonstrates missed loops on unfixed code
  - Test shows: Valid revisit at frame 600 (100 frames later) suppressed by 200-frame dead zone
  - Test shows: BoW candidate with score 0.0105 rejected by 0.012 threshold
  - Create synthetic scenarios with known loop closure opportunities
  - Document suppressed loops and rejected candidates
  - Expected outcome: Test confirms loops are suppressed (validates bug)
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 3. Write preservation property tests - Scale estimation
  - **Property 2: Preservation** - Diagnostic Modes and Edge Cases Unchanged
  - Follow observation-first methodology on UNFIXED code
  - Test diagnostic modes: scale_mode='gt', 'fixed', 'none' (observe behavior)
  - Test edge cases: empty map_points, non-finite pose, clean depths
  - Write property tests capturing observed behavior patterns
  - Verify tests PASS on unfixed code (baseline established)
  - Expected outcome: Tests PASS (confirms baseline behavior exists)
  - _Requirements: 1.3, 1.4, 3.1, 3.3_

- [ ] 4. Write preservation property tests - Loop detection
  - **Property 2: Preservation** - Loop Detection Logic Unchanged
  - Follow observation-first methodology on UNFIXED code
  - Test scenes without loops: no detections with either parameters
  - Test geometric verification: rejects candidates with <15 inliers
  - Test feature tracking: detection and matching counts unchanged
  - Test callback exception handling: graceful continuation
  - Verify tests PASS on unfixed code (baseline established)
  - Expected outcome: Tests PASS (confirms baseline behavior exists)
  - _Requirements: 2.3, 3.2, 3.3, 3.4_

- [ ] 5. Implement IQR-based outlier rejection in `_recover_scale()` (pipeline.py)
  - Add outlier detection logic to `_recover_scale()` function
  - Extract recent 200 map point depths, compute Q1, Q3, IQR
  - Define Tukey bounds: lower = Q1 - 1.5×IQR, upper = Q3 + 1.5×IQR
  - Filter depths to retain only those in [lower, upper]
  - Handle edge cases: <5 points → return 1.0, all outliers → return 1.0, non-finite → filter
  - Compute median ONLY on filtered depths
  - Apply clamping to [scale_clamp_min, scale_clamp_max] = [0.3m, 80.0m]
  - Preserve diagnostic modes: scale_mode='gt', 'fixed', 'none' unchanged
  - _Bug_Condition: Outlier depths corrupt median_
  - _Expected_Behavior: IQR filtering removes outliers, robust median estimate_
  - _Preservation: Diagnostic modes unchanged, edge cases return 1.0_
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 6. Implement loop detection parameter tuning (loop_detector.py)
  - Update LoopDetector.__init__() default parameters
  - Change min_loop_gap_frames from 200 to 100 frames (3.33s dead zone)
  - Change min_bow_score from 0.012 to 0.009 (accept marginal matches)
  - Keep all other parameters unchanged: min_geo_inliers=15, consistency=3, temporal_window=20
  - Geometric verification (PnP RANSAC, ≥15 inliers) prevents false positives
  - Add comments documenting parameter changes and rationale
  - _Bug_Condition: Conservative parameters suppress valid loops_
  - _Expected_Behavior: Relaxed parameters allow more loop detections_
  - _Preservation: Geometric verification, callback handling unchanged_
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 7. Verify bug condition exploration test now passes - Scale estimation
  - **Property 1: Expected Behavior** - Outlier-Robust Depth Estimation
  - **IMPORTANT**: Re-run the SAME test from task 1 on FIXED code (do NOT write new test)
  - Test from task 1 encodes expected behavior: IQR filtering removes outliers
  - When test passes on fixed code, confirms bug is resolved
  - Verify counterexamples from Phase 1 are now resolved
  - Expected outcome: Test PASSES on fixed code (bug fixed)
  - _Requirements: Expected Behavior Properties from design_

- [ ] 8. Verify bug condition exploration test now passes - Loop detection
  - **Property 1: Expected Behavior** - Enhanced Loop Detection
  - **IMPORTANT**: Re-run the SAME test from task 2 on FIXED code (do NOT write new test)
  - With min_loop_gap_frames=100, revisits within 100 frames are allowed
  - With min_bow_score=0.009, marginal BoW candidates are accepted and geometrically verified
  - When test passes on fixed code, confirms loop detection improvement
  - Expected outcome: Test PASSES on fixed code (loop detection enhanced)
  - _Requirements: Expected Behavior Properties from design_

- [ ] 9. Verify preservation tests still pass - Scale estimation
  - **Property 2: Preservation** - Scale Estimation Preservation
  - **IMPORTANT**: Re-run the SAME tests from task 3 on FIXED code (do NOT write new tests)
  - Diagnostic modes produce identical results (scale_mode='gt', 'fixed', 'none')
  - Edge cases still return 1.0 (no crashes)
  - Clean depths yield same median as before (inlier-only sets unaffected)
  - Expected outcome: Tests PASS on fixed code (no regressions)
  - _Requirements: Preservation Requirements from design_

- [ ] 10. Verify preservation tests still pass - Loop detection
  - **Property 2: Preservation** - Loop Detection Preservation
  - **IMPORTANT**: Re-run the SAME tests from task 4 on FIXED code (do NOT write new tests)
  - Scenes without loops: identical behavior with new parameters
  - Geometric verification (PnP RANSAC, ≥15 inliers) still enforced
  - Feature tracking statistics unchanged
  - Callback exception handling preserved
  - Expected outcome: Tests PASS on fixed code (no regressions)
  - _Requirements: Preservation Requirements from design_

- [ ] 11. Unit tests - Scale estimation robustness
  - [ ] 11.1 IQR filtering effectiveness: Verify Tukey fence removes outliers while retaining inliers
    - Input: Depths with varying outlier proportions (0%, 10%, 50%)
    - Assert: Outliers removed, inliers retained
    - Property: filtered_variance ≤ raw_variance
    - _Requirements: 1.1, 1.2_
  
  - [ ] 11.2 Median convergence: Verify filtered median converges to true distribution mean
    - Input: Large sample (n=1000) from normal distribution with outliers
    - Assert: Filtered median closer to true mean than raw median
    - Metric: MAE improvement ≥ 50%
    - _Requirements: 1.1_
  
  - [ ] 11.3 Edge case handling: Few points, all outliers, non-finite, negative depths
    - Test: <5 points → return 1.0 without crash
    - Test: All filtered as outliers → return 1.0
    - Test: NaN/inf depths → filtered without crash
    - Test: Negative depths → filtered without affecting inliers
    - _Requirements: 1.2, 1.3, 1.4_
  
  - [ ] 11.4 Diagnostic mode preservation: scale_mode='gt', 'fixed', 'none' identical to original
    - Test: GT scale mode produces identical recovery
    - Test: Fixed scale returns same value
    - Test: None mode returns 1.0
    - Compare statistics: Detection counts, tracking success rates unchanged
    - _Requirements: 1.3_
  
  - [ ] 11.5 Clamping applied correctly: Verify [0.3m, 80.0m] bounds enforced
    - Test: Filtered median = 150.0m → Result = 80.0m (clamped)
    - Property: Result ∈ [scale_clamp_min, scale_clamp_max]
    - _Requirements: 1.4_

- [ ] 12. Unit tests - Loop detection tuning
  - [ ] 12.1 Reduced dead zone: 100-frame window allows more loops than 200-frame window
    - Setup: Synthetic loops at frame separations 50, 100, 150, 200, 250
    - Assert: Frame 100 and 150 revisits detected with new params (previously suppressed)
    - Metric: loop_count(100-frame) > loop_count(200-frame)
    - _Requirements: 2.1_
  
  - [ ] 12.2 Lowered BoW threshold: 0.009 threshold accepts candidates rejected by 0.012
    - Setup: BoW candidates with scores [0.008, 0.0095, 0.011, 0.013]
    - Assert: With old threshold: 0.008 rejected; 0.011, 0.013 accepted
    - Assert: With new threshold: 0.0095, 0.011, 0.013 accepted
    - Metric: Acceptance count increases
    - _Requirements: 2.2_
  
  - [ ] 12.3 Geometric verification enforcement: ≥15 inlier requirement still enforced
    - Setup: Loop candidates with inlier counts 5, 10, 15, 20
    - Assert: 5, 10 inlier candidates rejected (below threshold)
    - Assert: 15, 20 inlier candidates accepted
    - Metric: No false positives from lowered BoW threshold
    - _Requirements: 2.3_
  
  - [ ] 12.4 Existing mechanisms: Callback handling, deduplication, temporal window
    - Test: Callback exceptions don't crash thread
    - Test: Same landmark doesn't fire multiple events
    - Test: Recent keyframes (±20 frames) excluded
    - _Requirements: 2.3, 3.2, 3.4_

- [ ] 13. Property-based tests
  - [ ] 13.1 Outlier rejection reduces variance: filtered_variance ≤ raw_variance for all outlier-contaminated sets
    - Generator: Random depths from N(μ=3.0, σ=0.2) with outlier injection
    - Assert: Property holds for 100+ generated samples
    - Rationale: Outlier filtering should reduce noise
    - _Requirements: 1.1_
  
  - [ ] 13.2 Median convergence: For clean samples, filtered_median ≈ raw_median
    - Generator: Depths from N(μ=3.0, σ=0.2) without outliers
    - Assert: |median(filtered) - median(raw)| < 0.01 for n≥100
    - Rationale: Inlier-only sets should have identical behavior
    - _Requirements: 1.1_
  
  - [ ] 13.3 Loop detection monotonicity: Relaxed parameters detect ≥ original parameters
    - Setup: Run detector with (min_gap=200, min_bow=0.012) and (min_gap=100, min_bow=0.009)
    - Assert: loop_count(relaxed) >= loop_count(original)
    - Rationale: Relaxed parameters should never reduce detections
    - _Requirements: 2.1, 2.2_
  
  - [ ] 13.4 Preservation under non-buggy inputs: For clean depths, filtered behavior = original
    - Generator: Depth sets in [Q1-1.5×IQR, Q3+1.5×IQR] range (no outliers)
    - Assert: median(filtered) == median(raw) for all generated samples
    - Rationale: Filtering shouldn't affect inlier-only sets
    - _Requirements: 1.3_

- [ ] 14. Integration tests
  - [ ] 14.1 Full KITTI 00 sequence test (4541 frames)
    - Baseline: Record APE (~197m), loop count, execution time on unfixed code
    - Apply both fixes (IQR + loop detection tuning)
    - Measure: APE, loop count, execution time on fixed code
    - Success criteria: APE < 20m (improvement ≥177m), increased loop events, no crashes
    - Metrics: APE, loop closures, BA residuals, PGO residuals
    - _Requirements: 3.1, 3.2, 3.3, 3.4, AC 2.4_
  
  - [ ] 14.2 Loop closure effects on pose graph optimization
    - Verify: PGO converges with increased loop frequency
    - Check: Optimization residuals reasonable (<1.0)
    - Verify: Poses corrected toward ground truth
    - Assert: No divergence or NaN values
    - _Requirements: 3.2_
  
  - [ ] 14.3 Diagnostic mode integration: Run with scale_mode='gt'
    - Verify: GT scale produces consistent trajectory
    - Verify: Feature tracking unchanged
    - Verify: BA and PGO work correctly with GT scale
    - _Requirements: 1.3, 3.1_

- [ ] 15. Checkpoint - Ensure all tests pass
  - [ ] 15.1 Unit tests: Scale estimation (5/5) + Loop detection (4/4) = 9/9 passing
    - All scale estimation tests passing
    - All loop detection tests passing
    - No failures or skipped tests
  
  - [ ] 15.2 Property-based tests: All 4/4 properties validated
    - Variance reduction property: PASS
    - Median convergence property: PASS
    - Loop detection monotonicity property: PASS
    - Preservation property: PASS
  
  - [ ] 15.3 Integration tests: All 3/3 components verified
    - KITTI 00 sequence: APE < 20m ✓
    - Loop closure optimization: Converged ✓
    - Diagnostic modes: Unchanged ✓
  
  - [ ] 15.4 Final validation: All acceptance criteria met
    - APE reduction: ≥177m (from ~197m to <20m) ✓
    - Loop events: Increased substantially ✓
    - Feature tracking: Unchanged ✓
    - Diagnostic modes: Preserved ✓
    - Edge cases: Handled gracefully ✓
    - No regressions: Confirmed ✓
    - _Requirements: AC 2.1, AC 2.2, AC 2.3, AC 2.4, AC 2.5_

---

## Task Dependency Graph

```json
{
  "waves": [
    {
      "name": "Phase 1: Exploration (Before Fix)",
      "description": "Run on unfixed code - expected to demonstrate bug",
      "tasks": [1, 2]
    },
    {
      "name": "Phase 2: Preservation (Before Fix)",
      "description": "Run on unfixed code - capture baseline behavior",
      "tasks": [3, 4]
    },
    {
      "name": "Phase 3: Implementation",
      "description": "Apply fixes to pipeline.py and loop_detector.py",
      "tasks": [5, 6]
    },
    {
      "name": "Phase 4: Verification (After Fix)",
      "description": "Run on fixed code - verify fix works and no regressions",
      "tasks": [7, 8, 9, 10, 11, 12, 13, 14, 15]
    }
  ]
}
```

---

## Notes

### Key Principles

**Exploration-First Methodology**:
- Write tests BEFORE implementing fix
- Run exploration tests on UNFIXED code (expect FAIL)
- Failure confirms bug exists, not test failure
- Prevents fixing test instead of code

**Preservation-First Methodology**:
- Write preservation tests BEFORE implementing fix
- Run on UNFIXED code (expect PASS)
- Establishes baseline behavior
- Prevents regressions after fix

**Property-Based Testing**:
- Stronger guarantees than manual unit tests
- Generates many test cases automatically
- Catches edge cases across input domain
- Validates universal properties ("for all inputs...")

### Critical Success Factors

1. **APE Reduction**: Must achieve <20m on KITTI 00 (from ~197m)
2. **Bug Confirmation**: Exploration tests must FAIL on unfixed code
3. **Preservation**: Preservation tests must PASS on both unfixed and fixed code
4. **Verification**: Re-run exploration tests on fixed code (now PASS)
5. **No Regressions**: All existing behavior unchanged for non-buggy inputs

### Configuration for Experiments

```python
# In VOConfig for tuning without code changes:
cfg.outlier_rejection_enabled = True           # Enable/disable IQR filtering
cfg.iqr_multiplier = 1.5                       # Tukey fence multiplier (default)
cfg.min_depths_for_robust_estimate = 5         # Minimum inliers (default)

cfg.loop_detector_min_gap_frames = 100         # Dead zone width (default)
cfg.loop_detector_min_bow_score = 0.009        # BoW threshold (default)
```

### Performance Expectations

- **IQR Overhead**: <1ms per frame (O(n log n) for n=200)
- **Loop Events**: 2-3x increase on KITTI 00
- **APE Improvement**: 197m → <20m (>99% reduction)
- **Total Processing Time**: Minimal overhead (<5%)

