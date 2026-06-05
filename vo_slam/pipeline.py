"""
pipeline.py
-----------
Visual Odometry pipeline: the main orchestrator.

Frame-to-frame flow
-------------------
 INIT  →  detect features on frame 0, store as Keyframe 0
 TRACK →  for each new frame:
            1. Detect & match features vs last keyframe (descriptor) or
               track via optical flow (LK) – configurable
            2. Estimate relative pose  (MotionEstimator)
            3. Scale recovery          (depth normalisation)
            4. Accumulate pose         (PoseGraph)
            5. Triangulate new points  (Triangulator)
            6. Keyframe check          (KeyframeSelector)
            7. Store stats             (VOStats)

VSLAM hooks
-----------
``on_new_keyframe``  – callback for loop-closure / bundle adjustment
``map_points``       – all triangulated points (feed to local BA)
``keyframes``        – all keyframes
"""

from __future__ import annotations
import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

from .camera       import CameraModel
from .features     import (DetectorType, MatcherType,
                           FeatureDetector, FeatureMatcher, FrameFeatures)
from .motion       import (MotionEstimator, PoseEstimate,
                           compose_pose, invert_pose)
from .triangulation import Triangulator, MapPoint
from .keyframe     import Keyframe, KeyframeSelector
from .covisibility import CovisibilityGraph  # Stage-2 module — not yet implemented


# ═══════════════════════════════════════════════════════════════════════ #
#  Enums & Config                                                         #
# ═══════════════════════════════════════════════════════════════════════ #

class TrackingMode(Enum):
    DESCRIPTOR   = auto()   # detect + match every frame
    OPTICAL_FLOW = auto()   # LK tracking, re-detect on keyframe


class VOState(Enum):
    NOT_INIT   = auto()
    OK         = auto()
    LOST       = auto()


@dataclass
class VOConfig:
    # Detector
    detector_type   : DetectorType  = DetectorType.ORB
    max_features    : int           = 2000
    grid_rows       : int           = 4
    grid_cols       : int           = 4
    # Matching / tracking
    matcher_type    : MatcherType   = MatcherType.BF_HAMMING
    ratio_thresh    : float         = 0.75
    tracking_mode   : TrackingMode  = TrackingMode.DESCRIPTOR
    # Motion
    ransac_thresh   : float         = 1.0
    ransac_prob     : float         = 0.999
    min_inliers     : int           = 20
    # Scale (monocular)
    scale_mode      : str           = 'median_depth'          # SAFE default — use 'median_depth' only after Bug1+2 verified
    fixed_scale     : float         = 1.0
    # Triangulation
    max_reproj_err  : float         = 2.0
    min_parallax_deg: float         = 1.0
    min_depth       : float         = 0.1
    max_depth       : float         = 200.0
    # Keyframe selection
    kf_min_parallax : float         = 2.0
    kf_max_feat_ratio: float        = 0.75
    kf_max_rot_deg  : float         = 15.0
    kf_min_frames   : int           = 3
    kf_max_frames   : int           = 20
    # General
    store_images    : bool          = False   # keep raw frames in Keyframe


# ═══════════════════════════════════════════════════════════════════════ #
#  Per-frame stats                                                        #
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


# ═══════════════════════════════════════════════════════════════════════ #
#  Pose Graph                                                             #
# ═══════════════════════════════════════════════════════════════════════ #

class PoseGraph:
    """
    Lightweight SE3 pose accumulator.
    Stores absolute poses T_world_cam for every processed frame.
    Ready to receive loop-closure corrections.
    """

    def __init__(self):
        self._poses: List[np.ndarray] = []   # T_world_cam per frame

    def add(self, T_world_cam: np.ndarray):
        self._poses.append(T_world_cam.copy())

    def update(self, frame_id: int, T_world_cam: np.ndarray):
        """Correct a pose (e.g. after loop closure)."""
        if frame_id < len(self._poses):
            self._poses[frame_id] = T_world_cam.copy()

    @property
    def poses(self) -> List[np.ndarray]:
        return self._poses

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) trajectory in world frame."""
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
    Monocular Visual Odometry pipeline.

    Usage
    -----
    vo = VisualOdometry(camera, config)
    for frame in frames:
        result = vo.process(frame)
        print(result.T_world_cam)   # camera pose in world

    VSLAM hooks
    -----------
    vo.on_new_keyframe = my_loop_closure_fn
    """

    def __init__(
        self,
        camera : CameraModel,
        config : Optional[VOConfig] = None,
    ):
        self.camera  = camera
        self.cfg     = config or VOConfig()
        self.covis_graph = CovisibilityGraph(min_shared=15)  # Stage-2

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
            camera        = camera,
            max_reproj_err= self.cfg.max_reproj_err,
            min_depth     = self.cfg.min_depth,
            max_depth     = self.cfg.max_depth,
            min_parallax  = self.cfg.min_parallax_deg,
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
        self.state         : VOState       = VOState.NOT_INIT
        self.frame_id      : int           = 0
        self.kf_id         : int           = 0
        self.keyframes     : List[Keyframe] = []
        self.map_points    : List[MapPoint] = []
        self.T_world_cam   : np.ndarray    = np.eye(4)  # current absolute pose

        self._last_kf       : Optional[Keyframe]     = None
        self._last_gray     : Optional[np.ndarray]   = None
        self._last_features : Optional[FrameFeatures] = None

        # ── VSLAM hook ───────────────────────────────────────────────── #
        self.on_new_keyframe: Optional[Callable[[Keyframe], None]] = None

        # ── Diagnostics ──────────────────────────────────────────────── #
        self.stats_history  : List[FrameStats] = []

    # ================================================================== #
    #  Public API                                                          #
    # ================================================================== #

    def process(
        self,
        img       : np.ndarray,
        timestamp : float = 0.0,
    ) -> FrameStats:
        """
        Process one frame. Returns FrameStats for this frame.
        Pose accessible via self.T_world_cam after each call.
        """
        t0   = time.perf_counter()
        gray = self._to_gray(img)
        stats = FrameStats(frame_id=self.frame_id)

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
        """Reset VO to uninitialized state."""
        self.state         = VOState.NOT_INIT
        self.frame_id      = 0
        self.kf_id         = 0
        self.keyframes     = []
        self.map_points    = []
        self.T_world_cam   = np.eye(4)
        self._last_kf      = None
        self._last_gray    = None
        self._last_features= None
        self.pose_graph    = PoseGraph()
        self.stats_history = []
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
    #  Private: Initialisation                                            #
    # ================================================================== #

    def _initialize(
        self,
        gray      : np.ndarray,
        img       : np.ndarray,
        timestamp : float,
        stats     : FrameStats,
    ) -> FrameStats:
        """Detect features on the first frame, store as KF 0."""
        feats = self.detector.detect_and_compute(gray)
        stats.num_detected = len(feats)

        if len(feats) < 10:
            return stats  # stay NOT_INIT

        self.T_world_cam = np.eye(4)
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
        self.covis_graph.add_keyframe(kf)  # Stage-2
        return stats

    # ================================================================== #
    #  Private: Tracking                                                  #
    # ================================================================== #

    def _track(
        self,
        gray      : np.ndarray,
        img       : np.ndarray,
        timestamp : float,
        stats     : FrameStats,
    ) -> FrameStats:

        kf = self._last_kf
        prev_feats = self._last_features

        # ── 1. Detect / track features ──────────────────────────────── #
        cur_feats = self.detector.detect_and_compute(gray)
        stats.num_detected = len(cur_feats)

        # Match against PREVIOUS frame for pure Odometry
        match_prev = self.matcher.match(prev_feats, cur_feats)
        stats.num_matched = len(match_prev)

        if len(match_prev) < self.cfg.min_inliers:
            self.state = VOState.LOST
            return stats

        # ── 2. Estimate relative pose (prev -> cur) ──────────────────── #
        pose: PoseEstimate = self.estimator.estimate(
            match_prev.pts_ref, match_prev.pts_cur
        )
        stats.num_inliers = pose.num_inliers
        stats.h_score     = pose.H_score

        if not pose.success:
            self.state = VOState.LOST
            return stats

        # ── 3. Scale recovery (monocular) ────────────────────────────── #
        scale = self._recover_scale(pose)

        # ── 4. Compose absolute pose ─────────────────────────────────── #
        R_cur_prev = pose.R.T
        t_cur_prev = -(pose.R.T @ (pose.t.ravel() * scale))

        # DEBUG: print first 5 frames to verify sign convention
        if self.frame_id < 5:
            print(f"  [frame {self.frame_id}] R_cur_prev diag = {R_cur_prev.diagonal().round(3)}")
            print(f"  [frame {self.frame_id}] t_cur_prev      = {t_cur_prev.round(4)}")
            print(f"  [frame {self.frame_id}] inliers         = {pose.num_inliers}")

        T_rel = np.eye(4)
        T_rel[:3, :3] = R_cur_prev
        T_rel[:3,  3] = t_cur_prev

        T_new = compose_pose(self.T_world_cam, T_rel)

        # ── Pose health check ────────────────────────────────────────── #
        # Reject frames where the composed pose contains NaN/Inf or an
        # implausibly large step (> 50 m per frame at 10 Hz = 500 m/s).
        MAX_STEP_M = 50.0
        step = np.linalg.norm(T_new[:3, 3] - self.T_world_cam[:3, 3])
        if not np.isfinite(T_new).all() or step > MAX_STEP_M:
            self.state = VOState.LOST
            return stats   # keep last good pose; do not update

        self.T_world_cam = T_new
        self.pose_graph.add(self.T_world_cam)

        # ── 5. Match against KEYFRAME for Triangulation ──────────────── #
        match_kf = self.matcher.match(kf.features, cur_feats)
        
        kf_pose: PoseEstimate = self.estimator.estimate(
            match_kf.pts_ref, match_kf.pts_cur
        )

        new_mps = []
        if kf_pose.success:
            inlier_mask        = kf_pose.inlier_mask
            inlier_ref         = match_kf.pts_ref[inlier_mask]
            inlier_cur         = match_kf.pts_cur[inlier_mask]
            inlier_idx_ref     = match_kf.idx_ref[inlier_mask]
            inlier_idx_cur     = match_kf.idx_cur[inlier_mask]

            T_kf_world  = kf.T_cam_world
            T_cur_world = invert_pose(self.T_world_cam)

            new_mps, _ = self.triangulator.triangulate(
                T_ref_world = T_kf_world,
                T_cur_world = T_cur_world,
                pts_ref     = inlier_ref,
                pts_cur     = inlier_cur,
                idx_ref     = inlier_idx_ref,
                idx_cur     = inlier_idx_cur,
                descriptors = kf.features.descriptors,
            )
            self.map_points.extend(new_mps)
            stats.num_map_pts = len(self.map_points)

            # ── 6. Keyframe check ────────────────────────────────────────── #
            do_kf, kf_reason = self.kf_selector.should_insert(
                last_kf     = kf,
                R_rel       = kf_pose.R,
                pts_ref     = inlier_ref,
                pts_cur     = inlier_cur,
                num_tracked = kf_pose.num_inliers,
            )
        else:
            # If KF tracking drops, insert a new KF to prevent losing track
            do_kf = True
            kf_reason = "lost_kf_track"

        stats.is_keyframe = do_kf
        stats.kf_reason   = kf_reason

        if do_kf:
            new_kf = Keyframe(
                frame_id    = self.frame_id,
                kf_id       = self.kf_id,
                T_world_cam = self.T_world_cam.copy(),
                features    = cur_feats,
                timestamp   = timestamp,
                map_points  = new_mps,
                image       = gray.copy() if self.cfg.store_images else None,
            )
            self.keyframes.append(new_kf)
            self._last_kf = new_kf
            self.kf_id   += 1

            if self.on_new_keyframe:
                self.on_new_keyframe(new_kf)
            self.covis_graph.add_keyframe(new_kf)  # Stage-2

        self._last_gray     = gray.copy()
        self._last_features = cur_feats
        self.state          = VOState.OK
        return stats

    # ================================================================== #
    #  Scale recovery                                                     #
    # ================================================================== #

    def _recover_scale(self, pose: PoseEstimate) -> float:
        """
        Monocular VO recovers only unit translation direction.
        Scale is estimated by keeping the median depth of the last map
        constant between consecutive frames (heuristic).

        Scale is clamped to [SCALE_MIN, SCALE_MAX] to prevent explosion.
        If the raw estimate falls outside this window the frame is treated
        as unreliable and unit scale is returned instead.
        """
        SCALE_MIN = 0.1
        SCALE_MAX = 10.0

        mode = self.cfg.scale_mode

        if mode == 'fixed':
            return self.cfg.fixed_scale

        if mode == 'none':
            return 1.0

        # 'median_depth': normalise so that new triangulated depth ≈ last median
        if self.map_points and mode == 'median_depth':
            T_cur_world = invert_pose(self.T_world_cam)
            depth = self.triangulator.compute_median_depth(
                self.map_points[-min(200, len(self.map_points)):],
                T_cur_world,
            )
            if SCALE_MIN < depth < SCALE_MAX:
                return depth   # clamped: safe to use
            # Outside window → depth estimate is unreliable; fall through to 1.0

        return 1.0

    # ================================================================== #
    #  Helpers                                                            #
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
        if self.stats_history:
            proc_times = [s.process_ms for s in self.stats_history[1:]]
            if proc_times:
                lines.append(f"  Avg process time : {np.mean(proc_times):.1f} ms "
                              f"({1000/np.mean(proc_times):.1f} fps)")
        return "\n".join(lines)