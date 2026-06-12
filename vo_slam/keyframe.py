"""
keyframe.py
-----------
Keyframe data structure and keyframe-selection policy.

A Keyframe is a selected frame whose pose and features are stored
in the map. The KeyframeSelector decides when a new keyframe
should be created based on:

  1. Parallax angle  – enough baseline for reliable triangulation
  2. Feature overlap – tracked features falling below threshold
  3. Rotation        – large rotation (feature tracking degrades)
  4. Fixed interval  – fallback timer
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .features import FrameFeatures
from .triangulation import MapPoint
from .motion import rotation_angle_deg


# ═══════════════════════════════════════════════════════════════════════ #
#  Keyframe                                                               #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class Keyframe:
    """
    A selected frame stored in the local map.

    Attributes
    ----------
    frame_id   : global monotonic frame counter
    kf_id      : keyframe index (subset of frame_id)
    timestamp  : seconds (optional, from dataset)
    T_world_cam: 4×4 SE3 – camera-to-world pose
    features   : detected keypoints + descriptors
    map_points : triangulated 3-D points associated with this KF
    image      : stored gray image (None if memory-constrained)
    """

    frame_id    : int
    kf_id       : int
    T_world_cam : np.ndarray                # 4×4 SE3
    features    : FrameFeatures
    timestamp   : float = 0.0
    map_points  : List[MapPoint] = field(default_factory=list)
    image       : Optional[np.ndarray] = None   # store for visualisation

    # Fast lookup cache: feature_index -> MapPoint
    _feat_to_mp : Dict[int, MapPoint] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.build_map_point_lookup()

    def build_map_point_lookup(self):
        """Build a fast lookup table from feature index to MapPoint."""
        self._feat_to_mp = {}
        for mp in self.map_points:
            idx = mp.obs.get(self.kf_id)
            if idx is not None:
                self._feat_to_mp[idx] = mp

    def get_map_point_at_feature(self, feat_idx: int) -> Optional[MapPoint]:
        """Resolve which map point was observed at this feature index."""
        # Try cache first
        mp = self._feat_to_mp.get(feat_idx)
        if mp is not None:
            return mp
        
        # Fallback: search (if cache was not updated after adding MPs)
        for mp in self.map_points:
            if mp.obs.get(self.kf_id) == feat_idx:
                self._feat_to_mp[feat_idx] = mp
                return mp
        return None

    @property
    def position(self) -> np.ndarray:
        """Camera centre in world frame (3,)."""
        return self.T_world_cam[:3, 3].copy()

    @property
    def R_world_cam(self) -> np.ndarray:
        return self.T_world_cam[:3, :3]

    @property
    def T_cam_world(self) -> np.ndarray:
        """Inverted pose (world → cam)."""
        T = self.T_world_cam   # Atomic read of the reference
        R = T[:3, :3]
        t = T[:3, 3:4]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3:4] = -R.T @ t
        return T_inv

    def num_map_points(self) -> int:
        return len(self.map_points)

    def __repr__(self) -> str:
        pos = self.position.round(2)
        return (f"Keyframe(kf_id={self.kf_id}, frame={self.frame_id}, "
                f"pos={pos}, MPs={self.num_map_points()})")


# ═══════════════════════════════════════════════════════════════════════ #
#  Keyframe Selector                                                      #
# ═══════════════════════════════════════════════════════════════════════ #

class KeyframeSelector:
    """
    Decides when to insert a new keyframe.

    Parameters
    ----------
    min_parallax_deg  : min median parallax between KF and current frame (°)
    max_feature_ratio : insert KF if tracked_features / kf_features < threshold
    max_rotation_deg  : insert KF on large rotation (°)
    min_frames        : minimum frames since last KF (avoid burst insertion)
    max_frames        : force KF after this many frames regardless
    """

    def __init__(
        self,
        min_parallax_deg  : float = 2.0,
        max_feature_ratio : float = 0.75,
        max_rotation_deg  : float = 15.0,
        min_frames        : int   = 3,
        max_frames        : int   = 20,
    ):
        self.min_parallax_deg  = min_parallax_deg
        self.max_feature_ratio = max_feature_ratio
        self.max_rotation_deg  = max_rotation_deg
        self.min_frames        = min_frames
        self.max_frames        = max_frames

        self._frames_since_kf  = 0

    def should_insert(
        self,
        last_kf       : Keyframe,
        R_rel         : np.ndarray,   # 3×3 relative rotation
        pts_ref       : np.ndarray,   # matched points in last KF
        pts_cur       : np.ndarray,   # matched points in current frame
        num_tracked   : int,
    ) -> tuple[bool, str]:
        """
        Returns (insert: bool, reason: str).
        """
        self._frames_since_kf += 1

        # Guard: too soon
        if self._frames_since_kf < self.min_frames:
            return False, "too_soon"

        # Force: too long without KF
        if self._frames_since_kf >= self.max_frames:
            self._frames_since_kf = 0
            return True, "max_frames"

        # Check rotation
        rot_deg = rotation_angle_deg(R_rel)
        if rot_deg > self.max_rotation_deg:
            self._frames_since_kf = 0
            return True, f"rotation={rot_deg:.1f}°"

        # Check feature survival ratio
        n_kf = len(last_kf.features)
        if n_kf > 0:
            ratio = num_tracked / n_kf
            if ratio < self.max_feature_ratio:
                self._frames_since_kf = 0
                return True, f"feature_ratio={ratio:.2f}"

        # Check parallax
        parallax = self._median_parallax(pts_ref, pts_cur)
        if parallax > self.min_parallax_deg:
            self._frames_since_kf = 0
            return True, f"parallax={parallax:.1f}°"

        return False, "none"

    def reset(self):
        self._frames_since_kf = 0

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _median_parallax(pts_ref: np.ndarray, pts_cur: np.ndarray) -> float:
        """
        Median pixel displacement as a proxy for parallax angle.
        Proper parallax angle needs depth; this is a fast approximation.
        """
        if len(pts_ref) == 0:
            return 0.0
        diff = pts_cur - pts_ref
        dists = np.linalg.norm(diff, axis=1)
        return float(np.median(dists))   # pixels – threshold tuned accordingly
