"""
Returns PoseEstimate where:
  R, t  =  T_{ref←cur}  (OpenCV recoverPose convention)
           i.e.  X_ref = R @ X_cur + t

Callers that need T_{cur←ref} for pose accumulation must invert:
  R_fwd = R.T
  t_fwd = -R.T @ t
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from .camera import CameraModel


@dataclass
class PoseEstimate:
    """Result of one relative-pose estimation."""
    R         : np.ndarray          # 3×3 rotation matrix
    t         : np.ndarray          # 3×1 unit translation
    inlier_mask: np.ndarray         # (N,) bool – which correspondences are inliers
    num_inliers: int
    method    : str                 # 'essential' | 'homography' | 'failed'
    H_score   : float = 0.0        # homography ratio (>0.45 = planar scene)

    @property
    def success(self) -> bool:
        return self.method != 'failed'

    def transform_matrix(self) -> np.ndarray:
        """Return 4×4 SE3 [R | t; 0 1]."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3]  = self.t.ravel()
        return T


class MotionEstimator:
    """
    Estimates the relative pose (R, t) between two frames.

    Parameters
    ----------
    camera          : CameraModel
    ransac_prob     : RANSAC confidence (0–1)
    ransac_thresh   : reprojection threshold in pixels for Essential Mat
    min_inliers     : reject estimate if fewer inliers
    homography_check: detect planar scenes and warn
    """

    def __init__(
        self,
        camera          : CameraModel,
        ransac_prob     : float = 0.999,
        ransac_thresh   : float = 1.0,
        min_inliers     : int   = 15,
        homography_check: bool  = True,
    ):
        self.camera           = camera
        self.ransac_prob      = ransac_prob
        self.ransac_thresh    = ransac_thresh
        self.min_inliers      = min_inliers
        self.homography_check = homography_check

    # ------------------------------------------------------------------ #
    #  Main entry                                                          #
    # ------------------------------------------------------------------ #

    def estimate(
        self,
        pts_ref: np.ndarray,   # (N, 2) pixels in reference frame
        pts_cur: np.ndarray,   # (N, 2) pixels in current frame
    ) -> PoseEstimate:
        """
        Estimate relative pose from N ≥ 5 point correspondences.
        Points should already be undistorted (or this method undistorts them).
        """
        assert pts_ref.shape == pts_cur.shape and pts_ref.ndim == 2

        N = len(pts_ref)
        if N < 5:
            return self._failed(N)

        pts_r = pts_ref.astype(np.float32)
        pts_c = pts_cur.astype(np.float32)

        # ── 1. Essential matrix ──────────────────────────────────────── #
        E, e_mask = cv2.findEssentialMat(
            pts_r, pts_c, self.camera.K,
            method      = cv2.RANSAC,
            prob        = self.ransac_prob,
            threshold   = self.ransac_thresh,
        )

        if E is None or e_mask is None:
            return self._failed(N)

        e_mask = e_mask.ravel().astype(bool)
        n_e    = int(e_mask.sum())

        if n_e < self.min_inliers:
            return self._failed(N)

        # ── 2. Recover pose (chirality check) ───────────────────────── #
        n_inliers, R, t, p_mask = cv2.recoverPose(
            E, pts_r, pts_c, self.camera.K, mask=e_mask.astype(np.uint8).copy()
        )
        p_mask = p_mask.ravel().astype(bool)

        if n_inliers < self.min_inliers:
            return self._failed(N)

        # ── 3. Optional homography check ────────────────────────────── #
        h_score = 0.0
        if self.homography_check and n_e > 8:
            h_score = self._homography_score(pts_r[e_mask], pts_c[e_mask])

        return PoseEstimate(
            R           = R,
            t           = t,
            inlier_mask = p_mask,
            num_inliers = int(p_mask.sum()),
            method      = 'essential',
            H_score     = h_score,
        )

    # ------------------------------------------------------------------ #
    #  Fundamental matrix fallback (when K is unavailable / approximate)  #
    # ------------------------------------------------------------------ #

    def estimate_fundamental(
        self,
        pts_ref: np.ndarray,
        pts_cur: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (F, E, inlier_mask) without recovering R, t.
        Useful for debugging or when K is not calibrated.
        """
        F, mask = cv2.findFundamentalMat(
            pts_ref, pts_cur,
            cv2.FM_RANSAC, self.ransac_thresh, self.ransac_prob,
        )
        E = self.camera.K.T @ F @ self.camera.K
        return F, E, mask.ravel().astype(bool)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _homography_score(pts_r: np.ndarray, pts_c: np.ndarray) -> float:
        """
        Ratio of homography inliers to essential-matrix inliers.
        High (>0.45) → planar scene; VO may be unreliable.
        """
        H, h_mask = cv2.findHomography(pts_r, pts_c, cv2.RANSAC, 3.0)
        if H is None or h_mask is None:
            return 0.0
        return float(h_mask.sum()) / len(pts_r)

    @staticmethod
    def _failed(N: int) -> PoseEstimate:
        return PoseEstimate(
            R            = np.eye(3),
            t            = np.zeros((3, 1)),
            inlier_mask  = np.zeros(N, dtype=bool),
            num_inliers  = 0,
            method       = 'failed',
        )


# ═══════════════════════════════════════════════════════════════════════ #
#  SE3 Pose utilities                                                     #
# ═══════════════════════════════════════════════════════════════════════ #

def compose_pose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Compose two 4×4 SE3 transforms: T_world_cur = T_world_ref @ T_ref_cur."""
    return T1 @ T2


def invert_pose(T: np.ndarray) -> np.ndarray:
    """Invert a 4×4 SE3 matrix efficiently."""
    R = T[:3, :3]
    t = T[:3, 3:4]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3:4] = -R.T @ t
    return T_inv


def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle (degrees) of the rotation represented by R."""
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def translation_norm(T: np.ndarray) -> float:
    """Euclidean norm of the translation part of a 4×4 SE3 matrix."""
    return float(np.linalg.norm(T[:3, 3]))
