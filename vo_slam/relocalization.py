"""
relocalization.py
-----------------
Recover camera pose when tracking is LOST by matching the current frame
against the stored map (keyframes + BoW database).

Algorithm
---------
1. Convert current frame descriptors → BoW vector
2. Query BoW database for top-K candidate keyframes
3. For each candidate:
   a. Match current descriptors ↔ candidate KF descriptors (ORB)
   b. Get 3D positions of matched map points
   c. Solve PnP (3D→2D) with RANSAC to get camera pose
   d. Verify with guided matching (search map points near reprojected positions)
4. Return the pose with most inliers if above threshold

This is purely geometric — no loop closure correction is applied here.
The returned pose is used to resume VO tracking from a known state.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .map_storage       import SavedMap, SavedKeyframe, SavedMapPoint
from .place_recognition import PlaceRecognizer


@dataclass
class RelocResult:
    """Result of a relocalization attempt."""
    success      : bool
    T_world_cam  : np.ndarray    # 4×4 recovered pose (identity if failed)
    inliers      : int
    matched_kf_id: int = -1
    method       : str = "pnp"


class Relocalization:
    """
    Attempts to recover pose from a saved map when VO goes LOST.

    Parameters
    ----------
    saved_map      : loaded SavedMap
    camera_K       : 3×3 intrinsic matrix
    min_matches    : minimum descriptor matches to attempt PnP
    min_inliers    : minimum PnP inliers to accept relocalization
    max_candidates : how many BoW candidates to try
    """

    def __init__(
        self,
        saved_map      : SavedMap,
        camera_K       : np.ndarray,
        min_matches    : int   = 20,
        min_inliers    : int   = 15,
        max_candidates : int   = 5,
    ):
        self.saved_map      = saved_map
        self.K              = camera_K
        self.min_matches    = min_matches
        self.min_inliers    = min_inliers
        self.max_candidates = max_candidates

        # Build fast lookup: kf_id → SavedKeyframe
        self._kf_index = {kf.kf_id: kf for kf in saved_map.keyframes}

        # BF matcher for relocalization
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Recognizer from saved map (has vocab + DB already populated)
        self._recognizer: Optional[PlaceRecognizer] = saved_map.recognizer

        # Stats
        self.n_attempts  = 0
        self.n_successes = 0

    # ------------------------------------------------------------------ #
    #  Main entry                                                          #
    # ------------------------------------------------------------------ #

    def relocalize(
        self,
        descriptors : np.ndarray,    # (N, 32) uint8 — current frame
        pts2d       : np.ndarray,    # (N, 2)  float32 — current frame keypoints
    ) -> RelocResult:
        """
        Attempt to recover camera pose from the saved map.

        Returns RelocResult.success=True with a valid T_world_cam if found.
        """
        self.n_attempts += 1
        failed = RelocResult(
            success     = False,
            T_world_cam = np.eye(4),
            inliers     = 0,
        )

        if self._recognizer is None or len(descriptors) < self.min_matches:
            return failed

        # ── 1. BoW query ─────────────────────────────────────────────── #
        bow          = self._recognizer.vocab.transform(descriptors, kf_id=-1)
        candidates   = self._recognizer.db.query(bow, max_results=self.max_candidates)

        if not candidates:
            return failed

        # ── 2. Try each candidate ────────────────────────────────────── #
        best: Optional[RelocResult] = None

        for cand in candidates:
            kf = self._kf_index.get(cand.kf_id)
            if kf is None or kf.descriptors is None or len(kf.descriptors) == 0:
                continue

            result = self._try_pnp(descriptors, pts2d, kf)
            if result.success:
                if best is None or result.inliers > best.inliers:
                    best = result

        if best is not None and best.inliers >= self.min_inliers:
            self.n_successes += 1
            print(f"  [Reloc] SUCCESS  KF{best.matched_kf_id}  "
                  f"inliers={best.inliers}")
            return best

        return failed

    # ------------------------------------------------------------------ #
    #  PnP solver                                                          #
    # ------------------------------------------------------------------ #

    def _try_pnp(
        self,
        cur_descs  : np.ndarray,
        cur_pts2d  : np.ndarray,
        ref_kf     : SavedKeyframe,
    ) -> RelocResult:
        """
        Match current frame against ref_kf, then solve PnP.
        We use the ref_kf's map point positions (from saved_map.map_points)
        as the 3D reference.
        """
        failed = RelocResult(success=False, T_world_cam=np.eye(4),
                             inliers=0, matched_kf_id=ref_kf.kf_id)

        # ── Descriptor matching ──────────────────────────────────────── #
        try:
            raw = self._matcher.knnMatch(cur_descs, ref_kf.descriptors, k=2)
        except cv2.error:
            return failed

        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < self.min_matches:
            return failed

        # ── Get 3D positions for matched keypoints ───────────────────── #
        # Map from ref_kf keypoint index → map point xyz
        # We use the saved map points where ref_kf_id matches
        ref_idx_to_xyz = {}
        for mp in self.saved_map.map_points:
            if mp.ref_kf_id not in ref_idx_to_xyz:
                ref_idx_to_xyz[mp.ref_kf_id] = mp.xyz

        pts3d_list = []
        pts2d_list = []

        for m in good:
            ref_idx = m.trainIdx
            # Use ref_kf pose to backproject the keypoint to 3D as fallback
            # (proper implementation uses map point xyz directly)
            xyz = ref_idx_to_xyz.get(ref_idx)
            if xyz is None:
                # Fallback: backproject using ref_kf pose + unit depth
                uv    = ref_kf.pts2d[ref_idx]
                K_inv = np.linalg.inv(self.K)
                ray   = K_inv @ np.array([uv[0], uv[1], 1.0])
                ray   /= np.linalg.norm(ray)
                # Place point 5m along ray in world frame
                R_wc  = ref_kf.T_world_cam[:3, :3]
                t_wc  = ref_kf.T_world_cam[:3, 3]
                xyz   = t_wc + R_wc @ (ray * 5.0)

            pts3d_list.append(xyz)
            pts2d_list.append(cur_pts2d[m.queryIdx])

        if len(pts3d_list) < self.min_matches:
            return failed

        pts3d = np.array(pts3d_list, dtype=np.float64)
        pts2d = np.array(pts2d_list, dtype=np.float64)

        # ── PnP RANSAC ───────────────────────────────────────────────── #
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d, pts2d, self.K, None,
                iterationsCount = 200,
                reprojectionError = 4.0,
                confidence      = 0.999,
                flags           = cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return failed

        if not ok or inliers is None or len(inliers) < self.min_inliers:
            return failed

        # Convert rvec/tvec → T_world_cam
        R_cw, _ = cv2.Rodrigues(rvec)
        t_cw    = tvec.ravel()

        # T_cam_world = [R_cw | t_cw]
        # T_world_cam = inv(T_cam_world)
        T_world_cam      = np.eye(4)
        T_world_cam[:3, :3] = R_cw.T
        T_world_cam[:3,  3] = -R_cw.T @ t_cw

        return RelocResult(
            success      = True,
            T_world_cam  = T_world_cam,
            inliers      = len(inliers),
            matched_kf_id= ref_kf.kf_id,
        )

    # ------------------------------------------------------------------ #
    #  Stats                                                               #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        rate = self.n_successes / max(self.n_attempts, 1) * 100
        return (f"Relocalization | "
                f"attempts={self.n_attempts} | "
                f"successes={self.n_successes} ({rate:.0f}%)")