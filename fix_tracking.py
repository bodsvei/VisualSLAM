import re

with open('vo_slam/pipeline.py', 'r') as f:
    lines = f.readlines()

out = []
in_track = False
track_indent = ""

for line in lines:
    if line.strip().startswith('def _track('):
        in_track = True
        track_indent = line[:len(line) - len(line.lstrip())]
        out.append(line)
        continue
    
    if in_track:
        if line.strip() == '' or line.startswith(track_indent + ' '):
            pass # Skip existing track body
        else:
            # End of _track
            new_track = """        kf = self._last_kf
        cur_feats = self.detector.detect_and_compute(gray)
        stats.num_detected = len(cur_feats)

        # ── 1. Frame-to-frame Tracking for pose ────────────────────── #
        match_prev = self.matcher.match(self._last_features, cur_feats)
        stats.num_matched = len(match_prev)

        if len(match_prev) < self.cfg.min_inliers:
            self.state = VOState.LOST
            self._lost_frames_count += 1
            if self._lost_frames_count >= 10:
                self.state = VOState.NOT_INIT
                self._lost_frames_count = 0
            return stats

        pose_prev = self.estimator.estimate(match_prev.pts_ref, match_prev.pts_cur)
        stats.num_inliers = pose_prev.num_inliers
        stats.h_score     = pose_prev.H_score

        if not pose_prev.success:
            self.state = VOState.LOST
            self._lost_frames_count += 1
            if self._lost_frames_count >= 10:
                self.state = VOState.NOT_INIT
                self._lost_frames_count = 0
            return stats

        # ── Scale recovery ─────────────────────────────────────────── #
        scale = self.cfg.fixed_scale

        # ── Pose accumulation ──────────────────────────────────────── #
        T_rp = np.eye(4)
        T_rp[:3, :3] = pose_prev.R
        T_rp[:3,  3] = pose_prev.t.flatten() * scale

        T_new = compose_pose(self.T_world_cam, invert_pose(T_rp))

        # Health check
        MAX_STEP_M = 50.0
        step = np.linalg.norm(T_new[:3, 3] - self.T_world_cam[:3, 3])
        if not np.isfinite(T_new).all() or step > MAX_STEP_M:
            self.state = VOState.LOST
            self._lost_frames_count += 1
            if self._lost_frames_count >= 10:
                self.state = VOState.NOT_INIT
                self._lost_frames_count = 0
            return stats

        self._lost_frames_count = 0
        self.T_world_cam = T_new
        self.pose_graph.add(self.T_world_cam)

        # ── 2. KF-to-current matching for triangulation & insertion ── #
        match_kf = self.matcher.match(kf.features, cur_feats)
        if len(match_kf) >= 8:
            pose_kf = self.estimator.estimate(match_kf.pts_ref, match_kf.pts_cur)
            R_rel = pose_kf.R if pose_kf.success else np.eye(3)
        else:
            pose_kf = None
            R_rel = np.eye(3)

        # ── Keyframe Decision ──────────────────────────────────────── #
        do_kf, kf_reason = self.kf_selector.should_insert(
            last_kf     = kf,
            R_rel       = R_rel,
            pts_ref     = match_kf.pts_ref,
            pts_cur     = match_kf.pts_cur,
            num_tracked = len(match_kf),
        )

        stats.is_keyframe = do_kf
        stats.kf_reason   = kf_reason

        if do_kf:
            new_mps = []
            if pose_kf is not None and pose_kf.success:
                inlier_mask        = pose_kf.inlier_mask
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
                    ref_kf_id   = kf.kf_id,
                    cur_kf_id   = self.kf_id,
                    idx_ref     = inlier_idx_ref,
                    idx_cur     = inlier_idx_cur,
                    descriptors = kf.features.descriptors,
                )
                self.map_points.extend(new_mps)
                kf.map_points.extend(new_mps)
                stats.num_map_pts = len(self.map_points)

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
            self.covis_graph.add_keyframe(new_kf) 
        else:
            stats.num_map_pts = len(self.map_points)

        self._last_gray     = gray.copy()
        self._last_features = cur_feats
        self.state          = VOState.OK
        return stats
"""
            out.append(new_track)
            out.append(line)
            in_track = False
    else:
        out.append(line)

with open('vo_slam/pipeline.py', 'w') as f:
    f.writelines(out)

