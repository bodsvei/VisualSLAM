"""
triangulation.py
----------------
Triangulate 3-D map points from two views.

Uses OpenCV's triangulatePoints (DLT, linear) with optional
non-linear refinement and reprojection-error filtering.

Output: MapPoint list and a validity mask.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .camera import CameraModel


# ═══════════════════════════════════════════════════════════════════════ #
#  Data container                                                         #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class MapPoint:
    """A single triangulated 3-D point."""
    xyz          : np.ndarray       # (3,) float64 – world coordinates
    ref_idx      : int              # index in reference frame's keypoints
    cur_idx      : int              # index in current  frame's keypoints
    reproj_err   : float = 0.0     # reprojection error (pixels)
    observations : int  = 2        # how many frames have seen it
    descriptor   : Optional[np.ndarray] = None   # attached descriptor (optional)

    def __repr__(self):
        return f"MapPoint(xyz={self.xyz.round(2)}, err={self.reproj_err:.2f}px)"


# ═══════════════════════════════════════════════════════════════════════ #
#  Triangulator                                                           #
# ═══════════════════════════════════════════════════════════════════════ #

class Triangulator:
    """
    Triangulate point correspondences given two camera poses.

    Parameters
    ----------
    camera         : CameraModel – intrinsics
    max_reproj_err : filter threshold in pixels
    min_depth      : discard points closer than this (camera units)
    max_depth      : discard points farther than this
    min_parallax   : minimum parallax angle in degrees for reliable triangulation
    """

    def __init__(
        self,
        camera         : CameraModel,
        max_reproj_err : float = 2.0,
        min_depth      : float = 0.1,
        max_depth      : float = 200.0,
        min_parallax   : float = 1.0,
    ):
        self.camera         = camera
        self.max_reproj_err = max_reproj_err
        self.min_depth      = min_depth
        self.max_depth      = max_depth
        self.min_parallax   = min_parallax

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def triangulate(
        self,
        T_ref_world : np.ndarray,   # 4×4 SE3 – reference camera pose (world→cam)
        T_cur_world : np.ndarray,   # 4×4 SE3 – current  camera pose  (world→cam)
        pts_ref     : np.ndarray,   # (N, 2) pixel coords in reference frame
        pts_cur     : np.ndarray,   # (N, 2) pixel coords in current frame
        idx_ref     : Optional[np.ndarray] = None,  # keypoint indices
        idx_cur     : Optional[np.ndarray] = None,
        descriptors : Optional[np.ndarray] = None,
    ) -> Tuple[List[MapPoint], np.ndarray]:
        """
        Returns (map_points, valid_mask).
        valid_mask is a boolean array of length N.
        """
        N = len(pts_ref)
        if N == 0:
            return [], np.zeros(0, dtype=bool)

        idx_ref = idx_ref if idx_ref is not None else np.arange(N)
        idx_cur = idx_cur if idx_cur is not None else np.arange(N)

        # Projection matrices P = K @ [R | t]
        P_ref = self.camera.K @ T_ref_world[:3]   # 3×4
        P_cur = self.camera.K @ T_cur_world[:3]   # 3×4

        # ── DLT triangulation ────────────────────────────────────────── #
        pts4d = cv2.triangulatePoints(
            P_ref,
            P_cur,
            pts_ref.T.astype(np.float64),
            pts_cur.T.astype(np.float64),
        )                                          # 4×N homogeneous

        # Normalise to 3-D
        w     = pts4d[3]
        valid_w = np.abs(w) > 1e-9
        xyz   = np.full((N, 3), np.nan)
        xyz[valid_w] = (pts4d[:3, valid_w] / w[valid_w]).T

        # ── Filters ──────────────────────────────────────────────────── #
        valid = valid_w.copy()
        valid &= self._depth_filter(xyz, T_ref_world, T_cur_world)
        valid &= self._parallax_filter(xyz, T_ref_world, T_cur_world)
        valid_indices = np.where(valid)[0]

        # ── Reprojection error ───────────────────────────────────────── #
        map_points: List[MapPoint] = []
        final_valid = np.zeros(N, dtype=bool)

        for i in valid_indices:
            err = self._reprojection_error(xyz[i], P_ref, pts_ref[i], P_cur, pts_cur[i])
            if err < self.max_reproj_err:
                desc = descriptors[idx_ref[i]] if descriptors is not None else None
                mp = MapPoint(
                    xyz        = xyz[i].copy(),
                    ref_idx    = int(idx_ref[i]),
                    cur_idx    = int(idx_cur[i]),
                    reproj_err = err,
                    descriptor = desc,
                )
                map_points.append(mp)
                final_valid[i] = True

        return map_points, final_valid

    def compute_median_depth(
        self,
        map_points  : List[MapPoint],
        T_cam_world : np.ndarray,
    ) -> float:
        """
        Median depth of map points in the given camera frame.
        Used for monocular scale recovery.
        """
        if not map_points:
            return 1.0
        R = T_cam_world[:3, :3]
        t = T_cam_world[:3, 3]
        depths = [float((R @ mp.xyz + t)[2]) for mp in map_points]
        return float(np.median(depths)) if depths else 1.0

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _depth_filter(
        self,
        xyz         : np.ndarray,    # (N, 3)
        T_ref_world : np.ndarray,
        T_cur_world : np.ndarray,
    ) -> np.ndarray:
        """Require positive depth in BOTH cameras within [min_depth, max_depth]."""
        def depths_in(T):
            R, t = T[:3, :3], T[:3, 3]
            z = (R @ xyz.T).T[:, 2] + t[2]
            return z

        z_ref = depths_in(T_ref_world)
        z_cur = depths_in(T_cur_world)
        return (
            (z_ref >= self.min_depth) & (z_ref <= self.max_depth) &
            (z_cur >= self.min_depth) & (z_cur <= self.max_depth)
        )

    def _parallax_filter(
        self,
        xyz         : np.ndarray,
        T_ref_world : np.ndarray,
        T_cur_world : np.ndarray,
    ) -> np.ndarray:
        """Filter points with too-small parallax angle (degenerate case)."""
        if self.min_parallax <= 0:
            return np.ones(len(xyz), dtype=bool)

        # Camera centres in world frame
        c_ref = -T_ref_world[:3, :3].T @ T_ref_world[:3, 3]
        c_cur = -T_cur_world[:3, :3].T @ T_cur_world[:3, 3]

        v_ref  = xyz - c_ref              # (N,3)
        v_cur  = xyz - c_cur

        norm_r = np.linalg.norm(v_ref, axis=1, keepdims=True)
        norm_c = np.linalg.norm(v_cur, axis=1, keepdims=True)
        eps    = 1e-9

        cos_angle = np.sum(v_ref * v_cur, axis=1) / (
            (norm_r.ravel() + eps) * (norm_c.ravel() + eps)
        )
        cos_angle = np.clip(cos_angle, -1, 1)
        angles_deg = np.degrees(np.arccos(cos_angle))

        return angles_deg >= self.min_parallax

    @staticmethod
    def _reprojection_error(
        xyz    : np.ndarray,
        P_ref  : np.ndarray,
        pt_ref : np.ndarray,
        P_cur  : np.ndarray,
        pt_cur : np.ndarray,
    ) -> float:
        """Symmetric reprojection error (average of both views)."""
        def reproject(P, X):
            h = P @ np.append(X, 1.0)
            return h[:2] / (h[2] + 1e-12)

        e1 = np.linalg.norm(reproject(P_ref, xyz) - pt_ref)
        e2 = np.linalg.norm(reproject(P_cur, xyz) - pt_cur)
        return float((e1 + e2) / 2.0)
