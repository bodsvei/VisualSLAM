"""
features.py
-----------
Feature detection, description, and matching.

Detectors  : ORB (default), SIFT, FAST+ORB
Matchers   : Brute-Force (L2 / Hamming) + FLANN
Filters    : Lowe's ratio test, cross-check, grid-based suppression
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════ #
#  Enums                                                                  #
# ═══════════════════════════════════════════════════════════════════════ #

class DetectorType(Enum):
    ORB        = auto()
    SIFT       = auto()
    FAST_ORB   = auto()   # FAST corners + ORB descriptors


class MatcherType(Enum):
    BF_HAMMING = auto()   # for binary descriptors (ORB)
    BF_L2      = auto()   # for float descriptors (SIFT)
    FLANN      = auto()   # approximate nearest-neighbour


# ═══════════════════════════════════════════════════════════════════════ #
#  Data containers                                                        #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class FrameFeatures:
    """Keypoints + descriptors for one frame."""
    keypoints   : List[cv2.KeyPoint]
    descriptors : np.ndarray          # (N, D)
    pts2d       : np.ndarray = field(init=False)   # (N, 2) float32

    def __post_init__(self):
        if self.keypoints:
            self.pts2d = np.array([kp.pt for kp in self.keypoints], dtype=np.float32)
        else:
            self.pts2d = np.empty((0, 2), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.keypoints)


@dataclass
class MatchResult:
    """Paired indices and coordinates after matching + filtering."""
    idx_ref     : np.ndarray   # (M,) indices into reference FrameFeatures
    idx_cur     : np.ndarray   # (M,) indices into current  FrameFeatures
    pts_ref     : np.ndarray   # (M, 2) pixel coords in reference frame
    pts_cur     : np.ndarray   # (M, 2) pixel coords in current  frame
    distances   : np.ndarray   # (M,) descriptor distances

    def __len__(self) -> int:
        return len(self.idx_ref)


# ═══════════════════════════════════════════════════════════════════════ #
#  Feature Detector / Descriptor                                          #
# ═══════════════════════════════════════════════════════════════════════ #

class FeatureDetector:
    """
    Wraps an OpenCV detector + optional grid-based uniform distribution.

    Parameters
    ----------
    detector_type : DetectorType
    max_features  : soft cap on keypoints per frame
    grid_rows, grid_cols : cells for adaptive suppression (0 = off)
    """

    def __init__(
        self,
        detector_type : DetectorType = DetectorType.ORB,
        max_features  : int  = 2000,
        grid_rows     : int  = 4,
        grid_cols     : int  = 4,
        fast_threshold: int  = 20,
        orb_scale     : float = 1.2,
        orb_levels    : int  = 8,
    ):
        self.detector_type = detector_type
        self.max_features  = max_features
        self.grid_rows     = grid_rows
        self.grid_cols     = grid_cols

        # --- build detector ---
        if detector_type == DetectorType.ORB:
            self._detector = cv2.ORB_create(
                nfeatures=max_features,
                scaleFactor=orb_scale,
                nlevels=orb_levels,
            )
            self._descriptor = None  # same object

        elif detector_type == DetectorType.SIFT:
            self._detector   = cv2.SIFT_create(nfeatures=max_features)
            self._descriptor = None

        elif detector_type == DetectorType.FAST_ORB:
            self._fast       = cv2.FastFeatureDetector_create(
                threshold=fast_threshold, nonmaxSuppression=True
            )
            self._descriptor = cv2.ORB_create(
                nfeatures=max_features,
                scaleFactor=orb_scale,
                nlevels=orb_levels,
            )
            self._detector   = None  # FAST has no compute()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def detect_and_compute(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> FrameFeatures:
        """Detect keypoints and compute descriptors from a gray image."""
        gray = self._to_gray(img)

        if self.detector_type == DetectorType.FAST_ORB:
            kps = self._fast.detect(gray, mask)
            kps = self._grid_suppress(kps, gray.shape)
            kps, descs = self._descriptor.compute(gray, kps)
        else:
            kps, descs = self._detector.detectAndCompute(gray, mask)
            kps = self._grid_suppress(kps, gray.shape)
            if kps:
                kps, descs = self._detector.compute(gray, kps) \
                    if hasattr(self._detector, 'compute') else (kps, descs)

        if descs is None or len(kps) == 0:
            return FrameFeatures(keypoints=[], descriptors=np.empty((0, 32), np.uint8))

        # cap total
        if len(kps) > self.max_features:
            kps   = kps[:self.max_features]
            descs = descs[:self.max_features]

        return FrameFeatures(keypoints=list(kps), descriptors=descs)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _grid_suppress(
        self,
        kps: List[cv2.KeyPoint],
        shape: Tuple[int, int],
    ) -> List[cv2.KeyPoint]:
        """Keep top-response keypoints uniformly distributed over a grid."""
        if self.grid_rows == 0 or self.grid_cols == 0 or not kps:
            return kps

        H, W = shape[:2]
        cell_h = H / self.grid_rows
        cell_w = W / self.grid_cols
        quota  = max(1, self.max_features // (self.grid_rows * self.grid_cols))

        grid: dict = {}
        for kp in sorted(kps, key=lambda k: -k.response):
            r = min(int(kp.pt[1] / cell_h), self.grid_rows - 1)
            c = min(int(kp.pt[0] / cell_w), self.grid_cols - 1)
            cell = grid.setdefault((r, c), [])
            if len(cell) < quota:
                cell.append(kp)

        return [kp for cell in grid.values() for kp in cell]


# ═══════════════════════════════════════════════════════════════════════ #
#  Feature Matcher                                                        #
# ═══════════════════════════════════════════════════════════════════════ #

class FeatureMatcher:
    """
    Matches descriptors between two FrameFeatures.

    Applies:
      1. kNN (k=2) matching
      2. Lowe's ratio test
      3. Optional symmetric (cross-check) filter
    """

    def __init__(
        self,
        matcher_type  : MatcherType = MatcherType.BF_HAMMING,
        ratio_thresh  : float = 0.75,
        cross_check   : bool  = False,
    ):
        self.ratio_thresh = ratio_thresh
        self.cross_check  = cross_check
        self._matcher     = self._build_matcher(matcher_type)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def match(self, ref: FrameFeatures, cur: FrameFeatures) -> MatchResult:
        """
        Match ref → cur.
        Returns MatchResult with M valid correspondences.
        """
        if len(ref) == 0 or len(cur) == 0:
            return self._empty_result()

        matches = self._matcher.knnMatch(ref.descriptors, cur.descriptors, k=2)
        good    = self._ratio_filter(matches)

        if self.cross_check and len(good) > 0:
            good = self._symmetric_filter(good, ref, cur)

        if not good:
            return self._empty_result()

        idx_r = np.array([m.queryIdx for m in good], dtype=np.int32)
        idx_c = np.array([m.trainIdx for m in good], dtype=np.int32)
        dists = np.array([m.distance for m in good], dtype=np.float32)

        return MatchResult(
            idx_ref  = idx_r,
            idx_cur  = idx_c,
            pts_ref  = ref.pts2d[idx_r],
            pts_cur  = cur.pts2d[idx_c],
            distances= dists,
        )

    def match_optical_flow(
        self,
        ref_gray: np.ndarray,
        cur_gray: np.ndarray,
        ref_pts : np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Lucas-Kanade optical flow tracking (alternative to descriptor matching).
        Returns (tracked_ref_pts, tracked_cur_pts, valid_mask).
        """
        lk_params = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            ref_gray, cur_gray,
            ref_pts.reshape(-1, 1, 2).astype(np.float32),
            None, **lk_params,
        )
        if cur_pts is None:
            return ref_pts[:0], ref_pts[:0], np.zeros(len(ref_pts), bool)

        mask = status.ravel() == 1
        return ref_pts[mask], cur_pts.reshape(-1, 2)[mask], mask

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_matcher(mt: MatcherType) -> cv2.DescriptorMatcher:
        if mt == MatcherType.BF_HAMMING:
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif mt == MatcherType.BF_L2:
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif mt == MatcherType.FLANN:
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6,
                                key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        raise ValueError(f"Unknown MatcherType: {mt}")

    def _ratio_filter(self, matches) -> List[cv2.DMatch]:
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_thresh * n.distance:
                    good.append(m)
        return good

    def _symmetric_filter(
        self,
        forward: List[cv2.DMatch],
        ref: FrameFeatures,
        cur: FrameFeatures,
    ) -> List[cv2.DMatch]:
        """Keep only matches that survive in both directions."""
        reverse_matches = self._matcher.knnMatch(cur.descriptors, ref.descriptors, k=2)
        reverse_good    = self._ratio_filter(reverse_matches)
        reverse_set     = {(m.trainIdx, m.queryIdx) for m in reverse_good}
        return [m for m in forward if (m.queryIdx, m.trainIdx) in reverse_set]

    @staticmethod
    def _empty_result() -> MatchResult:
        z = np.empty(0, dtype=np.int32)
        return MatchResult(z, z,
                           np.empty((0, 2), np.float32),
                           np.empty((0, 2), np.float32),
                           np.empty(0, np.float32))
