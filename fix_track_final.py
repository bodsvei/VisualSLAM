import re

with open('vo_slam/pipeline.py', 'r') as f:
    code = f.read()

# Define the CLEAN _track method body
clean_track_method = """    def _track(self, gray: np.ndarray, img: np.ndarray, timestamp: float, stats: FrameStats) -> FrameStats:
        kf = self._last_kf
        cur_feats = self.detector.detect_and_compute(gray)
        stats.num_detected = len(cur_feats)

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
        if self.cfg.scale_mode == 'fixed' or len(self.keyframes) < 5:
            # Bootstrap or Fixed mode
            scale = self.cfg.fixed_scale
            if self.cfg.scale_mode == 'fixed':
                # Pure frame-to-frame (smooth arcs, no bulging)
                T_rp = np.eye(4)
                T_rp[:3, :3] = pose_prev.R
                T_rp[:3,  3] = pose_prev.t.flatten() * scale
                T_new = compose_pose(self.T_world_cam, invert_pose(T_rp))
            else:
                # KF-relative bootstrap (ensures we get a baseline)
                if pose_kf.success:
                    T_rp = np.eye(4)
                    T_rp[:3, :3] = pose_kf.R
                    T_rp[:3,  3] = pose_kf.t.flatten() * (self.frame_id - kf.frame_id) * scale
                    T_new = compose_pose(kf.T_world_cam, invert_pose(T_rp))
                else:
                    T_rp = np.eye(4)
                    T_rp[:3, :3] = pose_prev.R
                    T_rp[:3,  3] = pose_prev.t.flatten() * scale
                    T_new = compose_pose(self.T_world_cam, invert_pose(T_rp))
        else:
            # Metric (median_depth): KF-relative is more stable for scale
            if pose_kf.success:
                scale = self._recover_scale(kf, match_kf, pose_kf.R, pose_kf.t)
                T_rp = np.eye(4)
                T_rp[:3, :3] = pose_kf.R
                T_rp[:3,  3] = pose_kf.t.flatten() * scale
                T_new = compose_pose(kf.T_world_cam, invert_pose(T_rp))
            else:
                # Fallback to frame-to-frame if KF matching fails
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
        self.pose_graph.add(self.T_world_cam)

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
"""

# Replace the messy _track method
pattern = r'    def _track\(self, gray: np.ndarray, img: np.ndarray, timestamp: float, stats: FrameStats\) -> FrameStats:.*?return stats'
code = re.sub(pattern, clean_track_method, code, flags=re.DOTALL)

with open('vo_slam/pipeline.py', 'w') as f:
    f.write(code)

