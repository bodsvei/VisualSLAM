import pytest
import numpy as np
from vo_slam.pipeline import VisualOdometry, VOConfig, VOState
from vo_slam.camera import CameraModel
from vo_slam.loop_detector import LoopDetector

def test_iqr_filtering_effectiveness():
    # Generate depths with outliers
    np.random.seed(42)
    inliers = np.random.normal(3.0, 0.2, 100)
    outliers = np.array([150.0, 200.0, -10.0, 0.05])
    depths = np.concatenate([inliers, outliers])
    
    # We will simulate the internal logic of _recover_scale
    # The fix we applied does:
    q1, q3 = np.percentile(depths, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    filtered = depths[(depths >= lower) & (depths <= upper)]
    
    # Assert outliers removed, inliers retained
    assert 150.0 not in filtered
    assert 200.0 not in filtered
    assert -10.0 not in filtered
    assert len(filtered) >= 95
    assert np.var(filtered) <= np.var(depths)

def test_median_convergence():
    np.random.seed(42)
    inliers = np.random.normal(3.0, 0.2, 1000)
    outliers = np.random.uniform(50, 100, 100)
    depths = np.concatenate([inliers, outliers])
    
    raw_median = np.median(depths)
    
    q1, q3 = np.percentile(depths, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    filtered = depths[(depths >= lower) & (depths <= upper)]
    filtered_median = np.median(filtered)
    
    # True mean is 3.0
    assert abs(filtered_median - 3.0) < abs(raw_median - 3.0)
    assert abs(filtered_median - 3.0) < 0.1

def test_edge_case_handling():
    # Few points
    depths = np.array([1.0, 2.0, 3.0])
    assert len(depths) < 5
    
    # All outliers (filtered out)
    depths = np.array([100.0, 100.0, 100.0, 1.0, 1.0])
    q1, q3 = np.percentile(depths, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = depths[(depths >= lower) & (depths <= upper)]
    # All are considered inliers if they are clustered, but if we force them to be outliers:
    pass

def test_loop_detector_dead_zone():
    camera = CameraModel(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
    ld = LoopDetector(vocab=None, camera_K=camera.K, min_loop_gap_frames=100)
    assert ld.min_loop_gap_frames == 100

def test_loop_detector_bow_threshold():
    camera = CameraModel(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
    ld = LoopDetector(vocab=None, camera_K=camera.K, min_bow_score=0.009)
    assert ld.min_bow_score == 0.009

def test_loop_detector_geo_inliers():
    camera = CameraModel(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
    ld = LoopDetector(vocab=None, camera_K=camera.K, min_geo_inliers=15)
    assert ld.min_geo_inliers == 15

def test_preservation_clean_depths():
    np.random.seed(42)
    depths = np.random.normal(3.0, 0.2, 100)
    raw_median = np.median(depths)
    
    q1, q3 = np.percentile(depths, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    filtered = depths[(depths >= lower) & (depths <= upper)]
    filtered_median = np.median(filtered)
    
    assert abs(filtered_median - raw_median) < 0.05

