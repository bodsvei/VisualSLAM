"""
camera.py
---------
Camera intrinsic model: pinhole + optional radial/tangential distortion.

Stores K (3x3), dist_coeffs (k1,k2,p1,p2[,k3]), image dimensions.
Provides undistortion, projection, and back-projection utilities.
"""

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CameraModel:
    """
    Pinhole camera with optional distortion.

    Attributes
    ----------
    fx, fy : focal lengths in pixels
    cx, cy : principal point in pixels
    width, height : image dimensions
    dist_coeffs : (k1, k2, p1, p2 [, k3]) OpenCV convention
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0   ],
        ], dtype=np.float64)

    @property
    def K_inv(self) -> np.ndarray:
        return np.linalg.inv(self.K)

    # ------------------------------------------------------------------ #
    #  Constructors                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_matrix(
        cls,
        K: np.ndarray,
        width: int,
        height: int,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> "CameraModel":
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=width,
            height=height,
            dist_coeffs=np.zeros(4) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64),
        )

    @classmethod
    def from_fov(
        cls,
        hfov_deg: float,
        width: int,
        height: int,
    ) -> "CameraModel":
        """Construct from horizontal field-of-view (square pixels assumed)."""
        hfov_rad = np.deg2rad(hfov_deg)
        fx = fy = (width / 2.0) / np.tan(hfov_rad / 2.0)
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)

    @classmethod
    def kitti(cls) -> "CameraModel":
        """KITTI sequence 00 camera 0 intrinsics."""
        K = np.array([
            [718.856, 0.0,     607.193],
            [0.0,     718.856, 185.216],
            [0.0,     0.0,     1.0    ],
        ], dtype=np.float64)
        return cls.from_matrix(K, width=1241, height=376)

    # ------------------------------------------------------------------ #
    #  Core operations                                                     #
    # ------------------------------------------------------------------ #

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        if np.allclose(self.dist_coeffs, 0):
            return img
        return cv2.undistort(img, self.K, self.dist_coeffs)

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Undistort Nx2 pixel points → Nx2 normalised image-plane coordinates.
        Returns pixel coords in the undistorted image (P=K kept).
        """
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        undist = cv2.undistortPoints(
            pts.reshape(-1, 1, 2).astype(np.float32),
            self.K,
            self.dist_coeffs,
            P=self.K,
        )
        return undist.reshape(-1, 2)

    def project(self, pts3d: np.ndarray) -> np.ndarray:
        """
        Project Nx3 world points (camera frame) → Nx2 pixel coordinates.
        Filters points behind the camera.
        """
        pts3d = np.asarray(pts3d, dtype=np.float64)
        if pts3d.ndim == 1:
            pts3d = pts3d.reshape(1, 3)
        z = pts3d[:, 2]
        valid = z > 0
        uv = np.full((len(pts3d), 2), np.nan)
        if valid.any():
            proj, _ = cv2.projectPoints(
                pts3d[valid].reshape(-1, 1, 3),
                np.zeros(3), np.zeros(3),
                self.K, self.dist_coeffs,
            )
            uv[valid] = proj.reshape(-1, 2)
        return uv

    def backproject(self, pts2d: np.ndarray, depth: float = 1.0) -> np.ndarray:
        """
        Lift Nx2 pixel coords to Nx3 rays (||ray||=1 in normalised plane).
        Optionally scale by `depth`.
        """
        pts2d = np.asarray(pts2d, dtype=np.float64)
        if pts2d.ndim == 1:
            pts2d = pts2d.reshape(1, 2)
        ones = np.ones((len(pts2d), 1))
        hom = np.hstack([pts2d, ones])          # Nx3
        rays = (self.K_inv @ hom.T).T           # Nx3
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return rays * depth

    def in_image(self, pts: np.ndarray, margin: int = 0) -> np.ndarray:
        """Boolean mask: which Nx2 pixel points are inside the image."""
        pts = np.asarray(pts)
        return (
            (pts[:, 0] >= margin) & (pts[:, 0] < self.width  - margin) &
            (pts[:, 1] >= margin) & (pts[:, 1] < self.height - margin)
        )

    def __repr__(self) -> str:
        return (
            f"CameraModel(fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"cx={self.cx:.2f}, cy={self.cy:.2f}, "
            f"{self.width}×{self.height})"
        )
