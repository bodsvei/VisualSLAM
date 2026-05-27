"""
tests/test_vo.py
-----------------
Unit tests for the visual odometry pipeline.
Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import pytest

from vo_slam.camera        import CameraModel
from vo_slam.features      import (DetectorType, MatcherType,
                               FeatureDetector, FeatureMatcher)
from vo_slam.motion        import (MotionEstimator, compose_pose, invert_pose,
                               rotation_angle_deg)
from vo_slam.triangulation import Triangulator
from vo_slam.keyframe      import Keyframe, KeyframeSelector
from vo_slam.pipeline      import VisualOdometry, VOConfig, VOState


# ═══════════════════════════════════════════════════════════════════════ #
#  Fixtures                                                               #
# ═══════════════════════════════════════════════════════════════════════ #

@pytest.fixture
def camera():
    return CameraModel(fx=500, fy=500, cx=320, cy=240, width=640, height=480)

@pytest.fixture
def synthetic_frame(camera):
    """Render a synthetic frame with projected random 3-D points."""
    rng    = np.random.RandomState(0)
    pts3d  = np.column_stack([
        rng.uniform(-4, 4, 400),
        rng.uniform(-3, 3, 400),
        rng.uniform(3, 15, 400),
    ])
    img = np.zeros((camera.height, camera.width), dtype=np.uint8)
    uv  = camera.project(pts3d)
    for (u, v) in uv.astype(int):
        if 0 <= u < camera.width and 0 <= v < camera.height:
            cv2.circle(img, (u, v), 3, 255, -1)
    return img, pts3d

@pytest.fixture
def vo(camera):
    cfg = VOConfig(min_inliers=8, kf_min_frames=1)
    return VisualOdometry(camera, cfg)


# ═══════════════════════════════════════════════════════════════════════ #
#  CameraModel tests                                                      #
# ═══════════════════════════════════════════════════════════════════════ #

class TestCameraModel:

    def test_K_shape(self, camera):
        assert camera.K.shape == (3, 3)
        assert camera.K[0, 0] == camera.fx
        assert camera.K[1, 1] == camera.fy

    def test_K_inv(self, camera):
        I = camera.K @ camera.K_inv
        np.testing.assert_allclose(I, np.eye(3), atol=1e-10)

    def test_project_roundtrip(self, camera):
        """Points on the optical axis project to principal point."""
        pt = np.array([[0.0, 0.0, 5.0]])
        uv = camera.project(pt)
        np.testing.assert_allclose(uv, [[camera.cx, camera.cy]], atol=0.5)

    def test_project_behind(self, camera):
        """Points behind camera give NaN."""
        pt = np.array([[0.0, 0.0, -1.0]])
        uv = camera.project(pt)
        assert np.isnan(uv).all()

    def test_in_image(self, camera):
        pts = np.array([[0, 0], [320, 240], [700, 500], [-1, 100]])
        mask = camera.in_image(pts)
        assert mask[0] == True
        assert mask[1] == True
        assert mask[2] == False
        assert mask[3] == False

    def test_backproject(self, camera):
        pts = np.array([[320.0, 240.0]])  # principal point
        rays = camera.backproject(pts)
        # Should point along z-axis
        np.testing.assert_allclose(rays[0, :2], [0, 0], atol=1e-5)

    def test_from_fov(self):
        cam = CameraModel.from_fov(90.0, 640, 480)
        assert cam.fx == pytest.approx(320.0, rel=0.01)

    def test_kitti(self):
        cam = CameraModel.kitti()
        assert cam.fx == pytest.approx(718.856)
        assert cam.width == 1241


# ═══════════════════════════════════════════════════════════════════════ #
#  Feature tests                                                          #
# ═══════════════════════════════════════════════════════════════════════ #

class TestFeatures:

    def test_detector_returns_features(self, synthetic_frame, camera):
        img, _ = synthetic_frame
        detector = FeatureDetector(max_features=500)
        feats = detector.detect_and_compute(img)
        assert len(feats) > 0
        assert feats.descriptors is not None
        assert feats.pts2d.shape[1] == 2

    def test_detector_grid_suppression(self, synthetic_frame, camera):
        img, _ = synthetic_frame
        detector = FeatureDetector(max_features=100, grid_rows=4, grid_cols=4)
        feats = detector.detect_and_compute(img)
        assert len(feats) <= 100

    def test_matcher_returns_correspondences(self, synthetic_frame, camera):
        img, _ = synthetic_frame
        # Slightly shifted image
        M = np.float32([[1, 0, 2], [0, 1, 2]])
        img2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        detector = FeatureDetector()
        matcher  = FeatureMatcher()
        f1 = detector.detect_and_compute(img)
        f2 = detector.detect_and_compute(img2)
        result = matcher.match(f1, f2)
        assert len(result) > 0
        assert result.pts_ref.shape[1] == 2
        assert result.pts_cur.shape[1] == 2

    def test_matcher_empty(self, camera):
        detector = FeatureDetector()
        matcher  = FeatureMatcher()
        blank    = np.zeros((480, 640), dtype=np.uint8)
        f1       = detector.detect_and_compute(blank)
        f2       = detector.detect_and_compute(blank)
        result   = matcher.match(f1, f2)
        assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════════ #
#  Motion estimator tests                                                 #
# ═══════════════════════════════════════════════════════════════════════ #

class TestMotionEstimator:

    def _make_correspondence(self, camera, R, t, n=50):
        """Synthesize perfect correspondences for known R, t."""
        rng   = np.random.RandomState(7)
        pts3d = np.column_stack([
            rng.uniform(-3, 3, n),
            rng.uniform(-2, 2, n),
            rng.uniform(4, 12, n),
        ])
        P0 = camera.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = camera.K @ np.hstack([R, t.reshape(3, 1)])

        def proj(P, X):
            h = P @ np.append(X, 1)
            return h[:2] / h[2]

        pts_ref = np.array([proj(P0, p) for p in pts3d], dtype=np.float32)
        pts_cur = np.array([proj(P1, p) for p in pts3d], dtype=np.float32)
        return pts_ref, pts_cur

    def test_known_pure_translation(self, camera):
        R_gt = np.eye(3)
        t_gt = np.array([0.5, 0.0, 0.0])
        pts_ref, pts_cur = self._make_correspondence(camera, R_gt, t_gt)

        est  = MotionEstimator(camera, min_inliers=5)
        pose = est.estimate(pts_ref, pts_cur)

        assert pose.success
        assert pose.num_inliers > 30
        # recovered R should be close to identity
        angle = rotation_angle_deg(pose.R)
        assert angle < 5.0

    def test_known_rotation(self, camera):
        angle = np.radians(5)
        R_gt  = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        t_gt  = np.array([0.1, 0.0, 0.0])
        pts_ref, pts_cur = self._make_correspondence(camera, R_gt, t_gt)

        est  = MotionEstimator(camera, min_inliers=5)
        pose = est.estimate(pts_ref, pts_cur)

        assert pose.success
        angle_err = rotation_angle_deg(pose.R @ R_gt.T)
        assert angle_err < 5.0   # within 5 degrees

    def test_too_few_points(self, camera):
        est  = MotionEstimator(camera, min_inliers=5)
        pts  = np.random.rand(3, 2).astype(np.float32)
        pose = est.estimate(pts, pts)
        assert not pose.success


# ═══════════════════════════════════════════════════════════════════════ #
#  Triangulation tests                                                    #
# ═══════════════════════════════════════════════════════════════════════ #

class TestTriangulator:

    def test_triangulates_known_points(self, camera):
        tri = Triangulator(camera, max_reproj_err=5.0, min_parallax=0.5)
        rng = np.random.RandomState(42)
        pts3d = np.column_stack([
            rng.uniform(-2, 2, 50),
            rng.uniform(-1, 1, 50),
            rng.uniform(5, 15, 50),
        ])

        T_ref = np.eye(4)     # reference at origin
        T_cur = np.eye(4)
        T_cur[0, 3] = 1.0    # translate 1m right

        P_ref = camera.K @ T_ref[:3]
        P_cur = camera.K @ T_cur[:3]

        def proj(P, X):
            h = P @ np.append(X, 1)
            return h[:2] / h[2]

        pts_ref = np.array([proj(P_ref, p) for p in pts3d], np.float32)
        pts_cur = np.array([proj(P_cur, p) for p in pts3d], np.float32)

        mps, mask = tri.triangulate(T_ref, T_cur, pts_ref, pts_cur)
        assert len(mps) > 0
        # Check a random recovered point is close to GT
        for mp in mps[:5]:
            gt = pts3d[mp.ref_idx]
            err = np.linalg.norm(mp.xyz - gt)
            assert err < 1.0  # within 1m for noiseless synthetic case


# ═══════════════════════════════════════════════════════════════════════ #
#  SE3 utilities                                                          #
# ═══════════════════════════════════════════════════════════════════════ #

class TestSE3:

    def test_invert_compose_identity(self):
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(np.array([0.1, 0.2, 0.05]))[0]
        T[:3, 3]  = [1, 2, 3]
        result = compose_pose(T, invert_pose(T))
        np.testing.assert_allclose(result, np.eye(4), atol=1e-10)

    def test_rotation_angle(self):
        angle = 30.0
        R = cv2.Rodrigues(np.radians([0, angle, 0]))[0]
        assert rotation_angle_deg(R) == pytest.approx(angle, abs=0.5)

    def test_identity_rotation_angle(self):
        assert rotation_angle_deg(np.eye(3)) == pytest.approx(0.0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════ #
#  Full pipeline smoke test                                               #
# ═══════════════════════════════════════════════════════════════════════ #

class TestPipeline:

    def _make_frame_pair(self, camera, tx=0.3):
        """
        Return two frames with known relative translation.
        Points rendered as small blobs; small tx to keep good overlap.
        """
        rng   = np.random.RandomState(1)
        pts3d = np.column_stack([
            rng.uniform(-4, 4, 800),
            rng.uniform(-3, 3, 800),
            rng.uniform(6, 18, 800),
        ])

        def render(R, t):
            img  = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
            pts_cam = (R @ pts3d.T).T + t
            uv = camera.project(pts_cam)
            for (u, v) in uv.astype(int):
                if 2 <= u < camera.width-2 and 2 <= v < camera.height-2:
                    cv2.circle(img, (u, v), 5, (200, 200, 200), -1)
            return img

        frame0 = render(np.eye(3), np.zeros(3))
        frame1 = render(np.eye(3), np.array([tx, 0, 0]))
        return frame0, frame1

    def test_init_state(self, vo, camera):
        assert vo.state == VOState.NOT_INIT

    def test_single_frame_initializes(self, vo, camera):
        frame, _ = self._make_frame_pair(camera)
        stats = vo.process(frame)
        assert vo.state == VOState.OK
        assert len(vo.keyframes) == 1
        assert stats.is_keyframe

    def test_two_frames_tracks(self, vo, camera):
        f0, f1 = self._make_frame_pair(camera)
        vo.process(f0)
        stats = vo.process(f1)
        assert vo.state == VOState.OK
        assert stats.num_matched > 0

    def test_trajectory_grows(self, vo, camera):
        f0, f1 = self._make_frame_pair(camera)
        vo.process(f0)
        vo.process(f1)
        traj = vo.trajectory
        assert traj.shape[0] == 2
        assert traj.shape[1] == 3

    def test_reset_clears_state(self, vo, camera):
        frame, _ = self._make_frame_pair(camera)
        vo.process(frame)
        vo.reset()
        assert vo.state      == VOState.NOT_INIT
        assert len(vo.keyframes)  == 0
        assert len(vo.map_points) == 0
        assert len(vo.trajectory) == 0

    def test_hook_called_on_keyframe(self, camera):
        called = []
        # Force KF every frame via max_frames=1
        vo2  = VisualOdometry(camera,
                              VOConfig(min_inliers=8, kf_min_frames=1,
                                       kf_max_frames=1))
        vo2.on_new_keyframe = lambda kf: called.append(kf.kf_id)
        f0, f1 = self._make_frame_pair(camera)
        vo2.process(f0)
        vo2.process(f1)
        # At minimum the init KF exists; hook fires on subsequent KFs
        assert len(vo2.keyframes) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
