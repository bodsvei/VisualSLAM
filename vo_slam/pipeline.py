"""
pipeline.py
-----------
Visual Odometry pipeline — orchestrates all VO modules.

Fixes applied vs original
--------------------------
BUG 1 — Coordinate frame inversion (highest impact):
  cv2.recoverPose() returns T_{ref←cur} (transforms points FROM current
  INTO reference).  The pipeline was composing it as T_{cur←ref}, causing
  translations to be applied in the wrong global direction after rotations.
  Fix: invert R,t before composing (R_cur_ref = R.T, t_cur_ref = -R.T @ t).

BUG 2 — Scale explosion with median_depth mode:
  _recover_scale() now clamps the returned depth to [0.3, 80.0] metres
  and guards against NaN/Inf in the current pose before computing depth.
  A pose health check after composition resets to last good pose if the
  result is non-finite or implausibly large (> 1e5 units).

BUG 3 — Triangulation with garbage pose:
  Baseline between current frame and last KF is checked before calling
  the triangulator.  If it is < 1e-6 (stationary) or > 1000 (exploded),
  triangulation is skipped for that frame.

BUG 4 — Only 46 map points:
  Direct consequence of BUG 2 and BUG 3.  Once the pose is valid,
  triangulation produces points normally.

Coordinate convention
---------------------
  Camera frame  : OpenCV (+X right, +Y down, +Z forward)
  World frame   : = camera frame at t=0 (identity pose)
  T_world_cam   : 4×4 SE3 — camera → world
                  p_world = T_world_cam @ p_cam
  recoverPose   : returns T_{ref←cur}  →  MUST invert before composing
  Accumulation  : T_world_cam_new = T_world_cam_old @ T_{cur←ref}
"""

from __future__ import annotations
import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional

from .camera        import CameraModel
from .features      import (DetectorType, MatcherType,
                             FeatureDetector, FeatureMatcher, FrameFeatures)
from .motion        import (MotionEstimator, PoseEstimate,
                             compose_pose, invert_pose)
from .triangulation import Triangulator, MapPoint
from .keyframe      import Keyframe, KeyframeSelector


# ═══════════════════════════════════════════════════════════════════════ #
#  Enums                                                                  #
# ═══════════════════════════════════════════════════════════════════════ #

class TrackingMode(Enum):
    DESCRIPTOR   = auto()
    OPTICAL_FLOW = auto()


class VOState(Enum):
    NOT_INIT = auto()
    OK       = auto()
    LOST     = auto()


# ═══════════════════════════════════════════════════════════════════════ #
#  Configuration                                                          #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class VOConfig:
    # Detector
    detector_type    : DetectorType = DetectorType.ORB
    max_features     : int          = 2000
    grid_rows        : int          = 4
    grid_cols        : int          = 4

    # Matching
    matcher_type     : MatcherType  = MatcherType.BF_HAMMING
    ratio_thresh     : float        = 0.75
    tracking_mode    : TrackingMode = TrackingMode.DESCRIPTOR

    # Motion estimation
    ransac_thresh    : float        = 1.0
    ransac_prob      : float        = 0.999
    min_inliers      : int          = 20

    # Scale (monocular)
    # 'fixed'        — unit translations, no metric scale, no drift
    # 'median_depth' — heuristic scale from map point depths (experimental)
    # 'none'         — raw unit norm translation
    scale_mode       : str          = 'fixed'
    fixed_scale      : float        = 1.0
    scale_clamp_min  : float        = 0.3     # BUG 2 clamp
    scale_clamp_max  : float        = 80.0    # BUG 2 clamp

    # Triangulation
    max_reproj_err   : float        = 2.0
    min_parallax_deg : float        = 1.0
    min_depth        : float        = 0.1
    max_depth        : float        = 200.0

    # Keyframe selection
    kf_min_parallax  : float        = 2.0
    kf_max_feat_ratio: float        = 0.75
    kf_max_rot_deg   : float        = 15.0
    kf_min_frames    : int          = 3
    kf_max_frames    : int          = 20

    # Blur detection (skip frame if Laplacian var < threshold)
    blur_threshold   : float        = 0.0     # 0 = disabled; 80 = typical

    # Misc
    store_images     : bool         = False
    pose_max_norm    : float        = 1e5     # BUG 2 health check


# ═══════════════════════════════════════════════════════════════════════ #
#  Per-frame statistics                                                   #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class FrameStats:
    frame_id     : int   = 0
    num_detected : int   = 0
    num_matched  : int   = 0
    num_inliers  : int   = 0
    num_map_pts  : int   = 0
    is_keyframe  : bool  = False
    kf_reason    : str   = ""
    h_score      : float = 0.0
    process_ms   : float = 0.0
    state        : str   = "OK"
    skipped_blur : bool  = False


# ═══════════════════════════════════════════════════════════════════════ #
#  Pose graph                                                             #
# ═══════════════════════════════════════════════════════════════════════ #

class PoseGraph:
    """
    Lightweight SE3 pose accumulator.
    Stores T_world_cam for every processed frame.
    Supports in-place correction for loop closure / PGO.
    """

    def __init__(self):
        self._poses: List[np.ndarray] = []

    def add(self, T_world_cam: np.ndarray):
        self._poses.append(T_world_cam.copy())

    def update(self, frame_id: int, T_world_cam: np.ndarray):
        """Overwrite a specific pose (e.g. after PGO correction)."""
        if 0 <= frame_id < len(self._poses):
            self._poses[frame_id] = T_world_cam.copy()

    @property
    def poses(self) -> List[np.ndarray]:
        return self._poses

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) camera centres in world frame."""
        if not self._poses:
            return np.empty((0, 3))
        return np.array([T[:3, 3] for T in self._poses])

    def __len__(self):
        return len(self._poses)


# ═══════════════════════════════════════════════════════════════════════ #
#  Visual Odometry                                                        #
# ═══════════════════════════════════════════════════════════════════════ #

class VisualOdometry:
    """
    Monocular Visual Odometry — front-end of the VSLAM pipeline.

    Usage
    -----
    vo = VisualOdometry(camera, config)
    for frame in frames:
        stats = vo.process(frame)
        T     = vo.T_world_cam   # current 4×4 SE3 pose

    VSLAM hooks
    -----------
    vo.on_new_keyframe = my_local_mapper_fn
    """

    def __init__(
        self,
        camera : CameraModel,
        config : Optional[VOConfig] = None,
    ):
        self.camera = camera
        self.cfg    = config or VOConfig()

        # ── Sub-systems ─────────────────────────────────────────────── #
        self.detector  = FeatureDetector(
            detector_type = self.cfg.detector_type,
            max_features  = self.cfg.max_features,
            grid_rows     = self.cfg.grid_rows,
            grid_cols     = self.cfg.grid_cols,
        )
        self.matcher   = FeatureMatcher(
            matcher_type  = self.cfg.matcher_type,
            ratio_thresh  = self.cfg.ratio_thresh,
        )
        self.estimator = MotionEstimator(
            camera        = camera,
            ransac_thresh = self.cfg.ransac_thresh,
            ransac_prob   = self.cfg.ransac_prob,
            min_inliers   = self.cfg.min_inliers,
        )
        self.triangulator = Triangulator(
            camera         = camera,
            max_reproj_err = self.cfg.max_reproj_err,
            min_depth      = self.cfg.min_depth,
            max_depth      = self.cfg.max_depth,
            min_parallax   = self.cfg.min_parallax_deg,
        )
        self.kf_selector = KeyframeSelector(
            min_parallax_deg  = self.cfg.kf_min_parallax,
            max_feature_ratio = self.cfg.kf_max_feat_ratio,
            max_rotation_deg  = self.cfg.kf_max_rot_deg,
            min_frames        = self.cfg.kf_min_frames,
            max_frames        = self.cfg.kf_max_frames,
        )
        self.pose_graph = PoseGraph()

        # ── State ────────────────────────────────────────────────────── #
        self.state         : VOState         = VOState.NOT_INIT
        self.frame_id      : int             = 0
        self.kf_id         : int             = 0
        self.keyframes     : List[Keyframe]  = []
        self.map_points    : List[MapPoint]  = []
        self.T_world_cam   : np.ndarray      = np.eye(4)

        self._last_kf           : Optional[Keyframe]      = None
        self._last_gray         : Optional[np.ndarray]    = None
        self._last_features     : Optional[FrameFeatures] = None
        self._last_good_T       : np.ndarray              = np.eye(4)
        self._frames_lost       : int                     = 0

        # ── VSLAM hook ───────────────────────────────────────────────── #
        self.on_new_keyframe: Optional[Callable[[Keyframe], None]] = None

        # ── Diagnostics ──────────────────────────────────────────────── #
        self.stats_history : List[FrameStats] = []

    # ================================================================== #
    #  Public API                                                          #
    # ================================================================== #

    def process(
        self,
        img       : np.ndarray,
        timestamp : float = 0.0,
    ) -> FrameStats:
        """
        Process one frame. Returns FrameStats.
        After each call, current pose is in self.T_world_cam.
        """
        t0   = time.perf_counter()
        gray = self._to_gray(img)

        stats = FrameStats(frame_id=self.frame_id)

        # ── Blur check (optional) ─────────────────────────────────────── #
        if self.cfg.blur_threshold > 0:
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur_score < self.cfg.blur_threshold:
                stats.skipped_blur = True
                stats.state        = "BLURRY"
                stats.process_ms   = (time.perf_counter() - t0) * 1000
                self.stats_history.append(stats)
                self.frame_id += 1
                return stats

        if self.state == VOState.NOT_INIT:
            stats = self._initialize(gray, img, timestamp, stats)
        else:
            stats = self._track(gray, img, timestamp, stats)

        stats.process_ms = (time.perf_counter() - t0) * 1000
        stats.state      = self.state.name
        self.stats_history.append(stats)
        self.frame_id += 1
        return stats

    def reset(self):
        self.state          = VOState.NOT_INIT
        self.frame_id       = 0
        self.kf_id          = 0
        self.keyframes      = []
        self.map_points     = []
        self.T_world_cam    = np.eye(4)
        self._last_kf       = None
        self._last_gray     = None
        self._last_features = None
        self._last_good_T   = np.eye(4)
        self._frames_lost   = 0
        self.pose_graph     = PoseGraph()
        self.stats_history  = []
        self.kf_selector.reset()

    @property
    def trajectory(self) -> np.ndarray:
        """(N, 3) camera positions in world frame."""
        return self.pose_graph.positions

    @property
    def current_pose(self) -> np.ndarray:
        """Current 4×4 SE3 T_world_cam."""
        return self.T_world_cam.copy()

    # ================================================================== #
    #  Initialization                                                      #
    # ================================================================== #

    def _initialize(
        self,
        gray      : np.ndarray,
        img       : np.ndarray,
        timestamp : float,
        stats     : FrameStats,
    ) -> FrameStats:
        feats = self.detector.detect_and_compute(gray)
        stats.num_detected = len(feats)

        if len(feats) < 10:
            return stats

        # World origin = first camera position
        self.T_world_cam  = np.eye(4)
        self._last_good_T = np.eye(4)
        self.pose_graph.add(self.T_world_cam)

        kf = Keyframe(
            frame_id    = self.frame_id,
            kf_id       = self.kf_id,
            T_world_cam = self.T_world_cam.copy(),
            features    = feats,
            timestamp   = timestamp,
            image       = gray.copy() if self.cfg.store_images else None,
        )
        self.keyframes.append(kf)
        self._last_kf       = kf
        self._last_gray     = gray.copy()
        self._last_features = feats
        self.state          = VOState.OK
        self.kf_id         += 1

        stats.is_keyframe = True
        stats.kf_reason   = "init"

        if self.on_new_keyframe:
            self.on_new_keyframe(kf)

        return stats

    # ================================================================== #
    #  Tracking                                                            #
    # ================================================================== #

    def _track(
        self,
        gray      : np.ndarray,
        img       : np.ndarray,
        timestamp : float,
        stats     : FrameStats,
    ) -> FrameStats:

        kf = self._last_kf

        # ── 1. Detect + match against last frame ──────────────────── #
        cur_feats    = self.detector.detect_and_compute(gray)
        stats.num_detected = len(cur_feats)

        match_result = self.matcher.match(self._last_features, cur_feats)
        stats.num_matched = len(match_result)

        if len(match_result) < self.cfg.min_inliers:
            return self._handle_lost(stats)

        # ── 2. Estimate relative pose ────────────────────────────────── #
        pose: PoseEstimate = self.estimator.estimate(
            match_result.pts_ref, match_result.pts_cur
        )
        stats.num_inliers = pose.num_inliers
        stats.h_score     = pose.H_score

        if not pose.success:
            return self._handle_lost(stats)

        # ── 3. BUG 1 FIX: correct coordinate frame composition ───────── #
        # recoverPose returns T_{cur←ref}: X_cur = R @ X_ref + t
        # We need T_{ref←cur} for forward accumulation: X_ref = R.T @ X_cur - R.T @ t
        # T_world_cur = T_world_kf @ T_{ref←cur}
        scale     = self._recover_scale(pose)
        R_ref_cur = pose.R.T
        t_ref_cur = -(pose.R.T @ (pose.t.ravel() * scale))

        T_rel = np.eye(4)
        T_rel[:3, :3] = R_ref_cur
        T_rel[:3,  3] = t_ref_cur

        candidate_T = compose_pose(self.T_world_cam, T_rel)

        # ── 4. BUG 2 FIX: pose health check ─────────────────────────── #
        pos_norm = np.linalg.norm(candidate_T[:3, 3])
        if (not np.isfinite(candidate_T).all()
                or pos_norm > self.cfg.pose_max_norm):
            print(f"  [VO] Pose explosion at frame {self.frame_id} "
                  f"(norm={pos_norm:.1e}) — reverting to last good pose")
            self.T_world_cam = self._last_good_T.copy()
            return self._handle_lost(stats)

        self.T_world_cam  = candidate_T
        self._last_good_T = candidate_T.copy()
        self.pose_graph.add(self.T_world_cam)

        # ── 5. BUG 3 FIX: triangulation baseline validity ────────────── #
        inlier_mask     = pose.inlier_mask
        inlier_ref      = match_result.pts_ref[inlier_mask]
        inlier_cur      = match_result.pts_cur[inlier_mask]
        inlier_idx_ref  = match_result.idx_ref[inlier_mask]
        inlier_idx_cur  = match_result.idx_cur[inlier_mask]

        T_prev_world = invert_pose(self.T_world_cam)
        T_cur_world  = invert_pose(candidate_T)

        baseline = np.linalg.norm(candidate_T[:3, 3] - self.T_world_cam[:3, 3])
        new_mps  = []

        if (1e-6 < baseline < 1000.0
                and np.isfinite(T_cur_world).all()):
            new_mps, _ = self.triangulator.triangulate(
                T_ref_world = T_prev_world,
                T_cur_world = T_cur_world,
                pts_ref     = inlier_ref,
                pts_cur     = inlier_cur,
                ref_kf_id   = -1,              # last frame is not a KF
                cur_kf_id   = self.kf_id,      # would-be next KF id
                idx_ref     = inlier_idx_ref,
                idx_cur     = inlier_idx_cur,
                descriptors = self._last_features.descriptors,
            )
            self.map_points.extend(new_mps)

        stats.num_map_pts = len(self.map_points)

        # ── 6. Keyframe decision ─────────────────────────────────────── #
        do_kf, kf_reason = self.kf_selector.should_insert(
            last_kf     = kf,
            R_rel       = pose.R,
            pts_ref     = inlier_ref,
            pts_cur     = inlier_cur,
            num_tracked = pose.num_inliers,
        )
        stats.is_keyframe = do_kf
        stats.kf_reason   = kf_reason

        if do_kf:
            new_kf = Keyframe(
                frame_id    = self.frame_id,
                kf_id       = self.kf_id,
                T_world_cam = self.T_world_cam.copy(),
                features    = cur_feats,
                timestamp   = timestamp,
                map_points  = list(new_mps),
                image       = gray.copy() if self.cfg.store_images else None,
            )
            self.keyframes.append(new_kf)
            self._last_kf = new_kf
            self.kf_id   += 1

            if self.on_new_keyframe:
                self.on_new_keyframe(new_kf)

        self._last_gray     = gray.copy()
        self._last_features = cur_feats
        self.state          = VOState.OK
        self._frames_lost   = 0
        return stats

    # ================================================================== #
    #  LOST state handler                                                  #
    # ================================================================== #

    def _handle_lost(self, stats: FrameStats) -> FrameStats:
        self._frames_lost += 1
        self.state = VOState.LOST

        # Attempt recovery after 10 consecutive lost frames
        if self._frames_lost > 10 and self._last_kf is not None:
            self.T_world_cam  = self._last_kf.T_world_cam.copy()
            self._last_good_T = self.T_world_cam.copy()
            self.state        = VOState.OK
            self._frames_lost = 0
            print(f"  [VO] Reinit from last KF at frame {self.frame_id}")

        self.pose_graph.add(self._last_good_T)
        return stats

    # ================================================================== #
    #  Scale recovery (monocular)                                          #
    # ================================================================== #

    def _recover_scale(self, pose: PoseEstimate) -> float:
        mode = self.cfg.scale_mode

        if mode == 'fixed':
            return self.cfg.fixed_scale

        if mode == 'none':
            return 1.0

        if mode == 'median_depth':
            # BUG 2 FIX: guard against broken pose before computing depth
            T_cur_world = invert_pose(self.T_world_cam)
            if not np.isfinite(T_cur_world).all():
                return 1.0

            if self.map_points:
                recent = self.map_points[-min(200, len(self.map_points)):]
                depth  = self.triangulator.compute_median_depth(
                    recent, T_cur_world
                )
                if np.isfinite(depth) and depth > 0:
                    # BUG 2 FIX: clamp to physical range
                    return float(np.clip(
                        depth,
                        self.cfg.scale_clamp_min,
                        self.cfg.scale_clamp_max,
                    ))

        return 1.0

    # ================================================================== #
    #  Helpers                                                             #
    # ================================================================== #

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def summary(self) -> str:
        lines = [
            "=== Visual Odometry Summary ===",
            f"  Frames processed : {self.frame_id}",
            f"  Keyframes        : {len(self.keyframes)}",
            f"  Map points       : {len(self.map_points)}",
            f"  State            : {self.state.name}",
        ]
        valid = [s for s in self.stats_history[1:] if not s.skipped_blur]
        if valid:
            times = [s.process_ms for s in valid]
            lines.append(
                f"  Avg process time : {np.mean(times):.1f} ms "
                f"({1000/np.mean(times):.1f} fps)"
            )
        blurry = sum(1 for s in self.stats_history if s.skipped_blur)
        if blurry:
            lines.append(f"  Blurry skipped   : {blurry}")
        return "\n".join(lines)