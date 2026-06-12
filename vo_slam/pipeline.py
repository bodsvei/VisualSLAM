"""
pipeline.py
-----------
Visual Odometry pipeline: the main orchestrator.
"""

from __future__ import annotations
import cv2
import numpy as np
import time
import threading
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
from .covisibility import CovisibilityGraph


class TrackingMode(Enum):
    DESCRIPTOR   = auto()
    OPTICAL_FLOW = auto()

class VOState(Enum):
    NOT_INIT   = auto()
    OK         = auto()
    LOST       = auto()

@dataclass
class VOConfig:
    detector_type   : DetectorType  = DetectorType.ORB
    max_features    : int           = 2000
    grid_rows       : int           = 4
    grid_cols       : int           = 4
    matcher_type    : MatcherType   = MatcherType.BF_HAMMING
    ratio_thresh    : float         = 0.75
    tracking_mode   : TrackingMode  = TrackingMode.DESCRIPTOR
    ransac_thresh   : float         = 1.0
    ransac_prob     : float         = 0.999
    min_inliers     : int           = 20
    scale_mode      : str           = 'metric'
    fixed_scale     : float         = 1.0     # fallback
    scale_clamp_min : float         = 0.3
    scale_clamp_max : float         = 80.0
    max_reproj_err  : float         = 2.0
    min_parallax_deg: float         = 1.0
    min_depth       : float         = 0.1
    max_depth       : float         = 200.0
    kf_min_parallax : float         = 1.5
    kf_max_feat_ratio: float        = 0.75
    kf_max_rot_deg  : float         = 15.0
    kf_min_frames   : int           = 3
    kf_max_frames   : int           = 20
    store_images    : bool          = False
    use_clahe       : bool          = True
    extrinsic_pitch : float         = 0.0     # Degrees, positive = tilted down

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

class PoseGraph:
    def __init__(self):
        self._poses: List[np.ndarray] = []
        self.lock = threading.Lock()

    def add(self, T_world_cam: np.ndarray):
        with self.lock:
            self._poses.append(T_world_cam.copy())

    def update(self, frame_id: int, T_world_cam: np.ndarray):
        with self.lock:
            if frame_id < len(self._poses):
                self._poses[frame_id] = T_world_cam.copy()

    @property
    def poses(self) -> List[np.ndarray]:
        with self.lock:
            return list(self._poses)

    @property
    def positions(self) -> np.ndarray:
        with self.lock:
            if not self._poses:
                return np.empty((0, 3))
            return np.array([T[:3, 3] for T in self._poses])

    def __len__(self):
        with self.lock:
            return len(self._poses)

    def transform_all(self, kfs: List[Keyframe], corrections: Dict[int, np.ndarray]):
        """Apply piece-wise constant corrections to the entire graph."""
        with self.lock:
            sorted_kfs = sorted(kfs, key=lambda k: k.frame_id)
            kf_idx = 0
            current_corr = np.eye(4)
            
            for fid in range(len(self._poses)):
                while kf_idx < len(sorted_kfs) and sorted_kfs[kf_idx].frame_id <= fid:
                    new_corr = corrections.get(sorted_kfs[kf_idx].kf_id)
                    if new_corr is not None:
                        current_corr = new_corr
                    kf_idx += 1
                
                if not np.array_equal(current_corr, np.eye(4)):
                    self._poses[fid] = current_corr @ self._poses[fid]


class VisualOdometry:
    def __init__(self, camera : CameraModel, config : Optional[VOConfig] = None):
        self.camera  = camera
        self.cfg     = config or VOConfig()
        self.covis_graph = CovisibilityGraph(min_shared=15)

        self.detector  = FeatureDetector(
            detector_type = self.cfg.detector_type,
            max_features  = self.cfg.max_features,
            grid_rows     = self.cfg.grid_rows,
            grid_cols     = self.cfg.grid_cols,
            use_clahe     = self.cfg.use_clahe,
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
        self.lock       = threading.Lock()

        self.state         : VOState       = VOState.NOT_INIT
        self.frame_id      : int           = 0
        self.kf_id         : int           = 0
        self.keyframes     : List[Keyframe] = []
        self.map_points    : List[MapPoint] = []
        self.T_world_cam   : np.ndarray    = np.eye(4)

        self._last_kf       : Optional[Keyframe]     = None
        self._last_gray     : Optional[np.ndarray]   = None
        self._last_features : Optional[FrameFeatures] = None
        self._lost_frames_count : int              = 0
        self._prev_scale    : float                = self.cfg.fixed_scale

        self.on_new_keyframe: Optional[Callable[[Keyframe], None]] = None
        self.stats_history  : List[FrameStats] = []

    def process(self, img: np.ndarray, img_right: Optional[np.ndarray] = None, timestamp: float = 0.0) -> FrameStats:
        t0   = time.perf_counter()
        gray = self._to_gray(img)
        gray_right = self._to_gray(img_right) if img_right is not None else None
        stats = FrameStats(frame_id=self.frame_id)

        with self.lock:
            if self.state == VOState.NOT_INIT:
                stats = self._initialize(gray, img, gray_right, timestamp, stats)
            else:
                stats = self._track(gray, img, gray_right, timestamp, stats)

            # Ensure pose graph always matches frame count
            if len(self.pose_graph) <= self.frame_id:
                self.pose_graph.add(self.T_world_cam)

        stats.process_ms = (time.perf_counter() - t0) * 1000
        stats.state      = self.state.name
        self.stats_history.append(stats)
        self.frame_id += 1
        return stats

    def reset(self):
        with self.lock:
            self.state         = VOState.NOT_INIT
            self.frame_id      = 0
            self.kf_id         = 0
            self.keyframes     = []
            self.map_points    = []
            self.T_world_cam   = np.eye(4)
            self._last_kf      = None
            self._last_gray    = None
            self._last_features= None
            self._lost_frames_count = 0
            self.pose_graph    = PoseGraph()
            self.stats_history = []
            self.kf_selector.reset()

    @property
    def trajectory(self) -> np.ndarray:
        return self.pose_graph.positions

    @property
    def current_pose(self) -> np.ndarray:
        with self.lock:
            return self.T_world_cam.copy()

    def _initialize(
        self, 
        gray        : np.ndarray, 
        img         : np.ndarray, 
        gray_right  : Optional[np.ndarray],
        timestamp   : float, 
        stats       : FrameStats
    ) -> FrameStats:
        feats = self.detector.detect_and_compute(gray)
        stats.num_detected = len(feats)

        if len(feats) < 10:
            return stats 

        if len(self.keyframes) == 0:
            # Initialize with identity (KITTI convention: world = first camera)
            self.T_world_cam = np.eye(4)
        else:
            # Re-initializing, keep the last known good pose
            pass

        # ── Stereo Metric Init ───────────────────────────────────────── #
        new_mps = []
        if gray_right is not None:
            feats_r = self.detector.detect_and_compute(gray_right)
            stereo_matches = self.matcher.match_stereo(feats, feats_r)
            if len(stereo_matches) > 10:
                new_mps = self.triangulator.triangulate_stereo(
                    feats, feats_r, stereo_matches, self.kf_id, self.T_world_cam
                )
                self.map_points.extend(new_mps)
                stats.num_map_pts = len(self.map_points)

        kf = Keyframe(
            frame_id    = self.frame_id,
            kf_id       = self.kf_id,
            T_world_cam = self.T_world_cam.copy(),
            features    = feats,
            timestamp   = timestamp,
            map_points  = new_mps,
            image       = gray.copy() if self.cfg.store_images else None,
        )
        self.keyframes.append(kf)
        self._last_kf       = kf
        self._last_gray     = gray.copy()
        self._last_features = feats
        self.state          = VOState.OK
        self.kf_id         += 1

        stats.is_keyframe = True
        stats.kf_reason   = "init_stereo" if gray_right is not None else "init"
        self.covis_graph.add_keyframe(kf)
        return stats

    def _track(
        self, 
        gray        : np.ndarray, 
        img         : np.ndarray, 
        gray_right  : Optional[np.ndarray],
        timestamp   : float, 
        stats       : FrameStats
    ) -> FrameStats:
        kf = self._last_kf
        cur_feats = self.detector.detect_and_compute(gray)
        stats.num_detected = len(cur_feats)

        # ── Stereo Depth Recovery (Current Frame) ───────────────────── #
        cur_depths = None
        if gray_right is not None:
            cur_feats_r = self.detector.detect_and_compute(gray_right)
            stereo_matches = self.matcher.match_stereo(cur_feats, cur_feats_r)
            if len(stereo_matches) > 10:
                # bf = baseline * fx
                disp = stereo_matches.pts_ref[:, 0] - stereo_matches.pts_cur[:, 0]
                valid = disp > 0.1
                if valid.any():
                    z = self.camera.bf / disp[valid]
                    cur_depths = {int(idx): float(depth) for idx, depth in zip(stereo_matches.idx_ref[valid], z)}

        # ── 1. Match against last frame for direction (robust) ─────── #
        match_prev = self.matcher.match(self._last_features, cur_feats)
        stats.num_matched = len(match_prev)

        if len(match_prev) < self.cfg.min_inliers:
            return self._handle_lost(stats)

        pose_prev = self.estimator.estimate(match_prev.pts_ref, match_prev.pts_cur)
        stats.num_inliers = pose_prev.num_inliers
        stats.h_score     = pose_prev.H_score

        if not pose_prev.success:
            return self._handle_lost(stats)

        # ── 2. Match against last Keyframe for scale/decision ──────── #
        match_kf = self.matcher.match(kf.features, cur_feats)
        pose_kf  = self.estimator.estimate(match_kf.pts_ref, match_kf.pts_cur)

        # ── 3. Tracking Logic Choice ────────────────────────────────── #
        use_stereo = (cur_depths is not None and self.cfg.scale_mode != 'fixed')

        if self.cfg.scale_mode == 'fixed':
            # Pure fixed-scale frame-to-frame
            scale = self.cfg.fixed_scale
            T_rp = np.eye(4)
            T_rp[:3, :3] = pose_prev.R
            T_rp[:3,  3] = pose_prev.t.flatten() * scale
            T_new = compose_pose(self.T_world_cam, invert_pose(T_rp))

        elif use_stereo or len(self.keyframes) >= 5:
            # Metric mode (Stereo or mature Monocular)
            if pose_kf.success:
                if cur_depths is not None:
                    # Robust metric scale from current stereo measurements
                    scale = self._recover_scale_stereo(match_kf, cur_depths, pose_kf.R, pose_kf.t)
                else:
                    # Fallback to monocular triangulation scale
                    scale = self._recover_scale(kf, match_kf, pose_kf.R, pose_kf.t)
                
                T_rp = np.eye(4)
                T_rp[:3, :3] = pose_kf.R
                T_rp[:3,  3] = pose_kf.t.flatten() * scale
                T_new = compose_pose(kf.T_world_cam, invert_pose(T_rp))
            else:
                # Fallback to frame-to-frame if KF matching fails
                scale = self._prev_scale
                T_rp = np.eye(4)
                T_rp[:3, :3] = pose_prev.R
                T_rp[:3,  3] = pose_prev.t.flatten() * scale
                T_new = compose_pose(self.T_world_cam, invert_pose(T_rp))
        
        else:
            # Monocular Bootstrap mode: KF-relative with fixed scale to get baseline
            if pose_kf.success:
                scale = self.cfg.fixed_scale
                T_rp = np.eye(4)
                T_rp[:3, :3] = pose_kf.R
                T_rp[:3,  3] = pose_kf.t.flatten() * (self.frame_id - kf.frame_id) * scale
                T_new = compose_pose(kf.T_world_cam, invert_pose(T_rp))
            else:
                scale = self.cfg.fixed_scale
                T_rp = np.eye(4)
                T_rp[:3, :3] = pose_prev.R
                T_rp[:3,  3] = pose_prev.t.flatten() * scale
                T_new = compose_pose(self.T_world_cam, invert_pose(T_rp))

        # ── 4. Health Check ─────────────────────────────────────────── #
        MAX_STEP_M = 50.0
        step = np.linalg.norm(T_new[:3, 3] - self.T_world_cam[:3, 3])
        
        if not np.isfinite(T_new).all() or step > MAX_STEP_M:
            return self._handle_lost(stats)

        self.T_world_cam = T_new
        self._lost_frames_count = 0

        # ── 5. Keyframe Decision (Accumulated Motion) ──────────────── #
        T_kf_cur = compose_pose(invert_pose(kf.T_world_cam), self.T_world_cam)
        R_rel_kf = T_kf_cur[:3, :3]
        
        do_kf, kf_reason = self.kf_selector.should_insert(
            last_kf     = kf,
            R_rel       = R_rel_kf,
            pts_ref     = match_kf.pts_ref,
            pts_cur     = match_kf.pts_cur,
            num_tracked = len(match_kf),
        )

        stats.is_keyframe = do_kf
        stats.kf_reason   = kf_reason

        if do_kf:
            # ── MapPoint Reuse (Covisibility) ────────────────────── #
            matched_mps = []
            if match_kf is not None:
                for i in range(len(match_kf)):
                    idx_ref = match_kf.idx_ref[i]
                    idx_cur = match_kf.idx_cur[i]
                    mp = kf.get_map_point_at_feature(idx_ref)
                    if mp is not None:
                        mp.obs[self.kf_id] = int(idx_cur)
                        matched_mps.append(mp)

            # ── New Triangulation ────────────────────────────── #
            new_mps = []
            if pose_kf is not None and pose_kf.success:
                inlier_mask = pose_kf.inlier_mask
                tri_indices = []
                for i in range(len(match_kf)):
                    if inlier_mask[i]:
                        idx_ref = match_kf.idx_ref[i]
                        if kf.get_map_point_at_feature(idx_ref) is None:
                            tri_indices.append(i)
                
                if tri_indices:
                    tri_indices = np.array(tri_indices)
                    inlier_ref         = match_kf.pts_ref[tri_indices]
                    inlier_cur         = match_kf.pts_cur[tri_indices]
                    inlier_idx_ref     = match_kf.idx_ref[tri_indices]
                    inlier_idx_cur     = match_kf.idx_cur[tri_indices]

                    T_kf_world  = kf.T_cam_world
                    T_cur_world = invert_pose(self.T_world_cam)

                    new_mps, _ = self.triangulator.triangulate(
                        T_ref_world = T_kf_world,
                        T_cur_world = T_cur_world,
                        pts_ref     = inlier_ref,
                        pts_cur     = inlier_cur,
                        ref_kf_id   = kf.kf_id,
                        cur_kf_id   = self.kf_id,
                        idx_ref     = inlier_idx_ref,
                        idx_cur     = inlier_idx_cur,
                        descriptors = kf.features.descriptors,
                    )
                    self.map_points.extend(new_mps)
                    stats.num_map_pts = len(self.map_points)

            all_kf_mps = matched_mps + new_mps

            # ── Additional Stereo Points (Current KF) ────────────────── #
            if gray_right is not None and cur_depths is not None:
                existing_idx = {mp.obs[self.kf_id] for mp in all_kf_mps if self.kf_id in mp.obs}
                
                stereo_mps = []
                for idx, depth in cur_depths.items():
                    if idx not in existing_idx:
                        z = depth
                        pt2d = cur_feats.pts2d[idx]
                        x = (pt2d[0] - self.camera.cx) * z / self.camera.fx
                        y = (pt2d[1] - self.camera.cy) * z / self.camera.fy
                        p_world = self.T_world_cam[:3, :3] @ np.array([x, y, z]) + self.T_world_cam[:3, 3]
                        mp = MapPoint(
                            xyz        = p_world,
                            ref_idx    = -1,
                            cur_idx    = int(idx),
                            reproj_err = 0.0,
                            descriptor = cur_feats.descriptors[idx],
                        )
                        mp.obs[self.kf_id] = int(idx)
                        stereo_mps.append(mp)
                
                self.map_points.extend(stereo_mps)
                all_kf_mps += stereo_mps
                stats.num_map_pts = len(self.map_points)

            new_kf = Keyframe(
                frame_id    = self.frame_id,
                kf_id       = self.kf_id,
                T_world_cam = self.T_world_cam.copy(),
                features    = cur_feats,
                timestamp   = timestamp,
                map_points  = all_kf_mps,
                image       = gray.copy() if self.cfg.store_images else None,
            )
            self.keyframes.append(new_kf)
            self._last_kf = new_kf
            self.kf_id   += 1

            if self.on_new_keyframe:
                self.on_new_keyframe(new_kf)
            self.covis_graph.add_keyframe(new_kf) 
        else:
            stats.num_map_pts = len(self.map_points)

        self._last_gray     = gray.copy()
        self._last_features = cur_feats
        self.state          = VOState.OK
        return stats


    def _handle_lost(self, stats: FrameStats) -> FrameStats:
        self.state = VOState.LOST
        self._lost_frames_count += 1
        if self._lost_frames_count >= 10:
            self.state = VOState.NOT_INIT
            self._lost_frames_count = 0
        
        return stats

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def summary(self) -> str:
        with self.lock:
            return f"VO Summary | Frames: {self.frame_id} | KFs: {len(self.keyframes)} | State: {self.state.name}"

    def _recover_scale_stereo(self, match_kf, cur_depths: dict, R: np.ndarray, t: np.ndarray) -> float:
        """
        Estimate metric scale using current stereo depth measurements.
        Scale s = median(Z_stereo / Z_triangulated_unscaled)
        """
        pts_ref_2d = match_kf.pts_ref
        pts_cur_2d = match_kf.pts_cur
        
        # Triangulate unscaled
        P1 = self.camera.K @ np.eye(3, 4)
        T_rel = np.eye(4); T_rel[:3,:3]=R; T_rel[:3,3]=t.ravel()
        P2 = self.camera.K @ T_rel[:3]
        
        pts4d = cv2.triangulatePoints(P1, P2, pts_ref_2d.T, pts_cur_2d.T)
        z_tri = pts4d[2] / (pts4d[3] + 1e-9)
        
        # Match with stereo depths
        ratios = []
        for i in range(len(match_kf)):
            idx_cur = int(match_kf.idx_cur[i])
            z_stereo = cur_depths.get(idx_cur)
            if z_stereo is not None and z_tri[i] > 0.1:
                ratios.append(z_stereo / z_tri[i])
                
        if not ratios:
            return self.cfg.fixed_scale
            
        return float(np.median(ratios))

    def _recover_scale(self, kf: Keyframe, match_kf, R: np.ndarray, t: np.ndarray) -> float:
        """
        Estimate relative scale between last KF and current frame.
        Uses depth ratios of triangulated MapPoints.
        """
        if self.cfg.scale_mode == 'fixed':
            return self.cfg.fixed_scale
        if self.cfg.scale_mode == 'none':
            return 1.0

        mps_ref = []
        pts_cur = []
        for i in range(len(match_kf)):
            idx_ref = match_kf.idx_ref[i]
            mp = kf.get_map_point_at_feature(idx_ref)
            if mp is not None:
                mps_ref.append(mp)
                pts_cur.append(match_kf.pts_cur[i])

        if len(mps_ref) < 5:
            return self.cfg.fixed_scale

        T_ref_world = kf.T_cam_world
        depths_ref = []
        for mp in mps_ref:
            p_cam = T_ref_world[:3, :3] @ mp.xyz + T_ref_world[:3, 3]
            depths_ref.append(p_cam[2])

        pts_ref_2d = np.array([kf.features.pts2d[mp.obs[kf.kf_id]] for mp in mps_ref], dtype=np.float32)
        pts_cur_2d = np.array(pts_cur, dtype=np.float32)

        P1 = self.camera.K @ np.eye(3, 4)
        T_rel = np.eye(4); T_rel[:3,:3]=R; T_rel[:3,3]=t.ravel()
        P2 = self.camera.K @ T_rel[:3]

        pts4d = cv2.triangulatePoints(P1, P2, pts_ref_2d.T, pts_cur_2d.T)
        depths_cur_unscaled = pts4d[2] / (pts4d[3] + 1e-9)

        valid = (depths_cur_unscaled > 0.1) & (np.array(depths_ref) > 0.1)
        if np.sum(valid) < 5:
            return self.cfg.fixed_scale

        ratios = np.array(depths_ref)[valid] / depths_cur_unscaled[valid]
        return float(np.clip(np.median(ratios), self.cfg.scale_clamp_min, self.cfg.scale_clamp_max))
