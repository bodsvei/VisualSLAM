import re

with open('vo_slam/pipeline.py', 'r') as f:
    content = f.read()

old_block = """        # --- Attempt Frame-to-Frame Tracking ---
        if len(match_prev) >= self.cfg.min_inliers:
            pose_prev = self.estimator.estimate(match_prev.pts_ref, match_prev.pts_cur)
            stats.num_inliers = pose_prev.num_inliers
            stats.H_score     = pose_prev.H_score 

            if pose_prev.success:
                # Calculate T_new using frame-to-frame logic (existing scale modes)
                scale = self.cfg.fixed_scale # Default scale
                if self.cfg.scale_mode == 'fixed':
                    scale = self.cfg.fixed_scale
                elif use_stereo or len(self.keyframes) >= 5: # Metric mode
                    if pose_kf.success:
                        # Bug 1 fix: Scale recovery expects T_{cur <- ref} 
                        # but estimator returns T_{ref <- cur}. Must invert.
                        T_cur_kf = invert_pose(pose_kf.transform_matrix_ref_from_cur())
                        if cur_depths is not None:
                            scale = self._recover_scale_stereo(match_kf, cur_depths, T_cur_kf[:3,:3], T_cur_kf[:3,3])
                        else:
                            scale = self._recover_scale(kf, match_kf, T_cur_kf[:3,:3], T_cur_kf[:3,3])
                    else:
                        scale = self._prev_scale
                else: # Monocular Bootstrap mode
                    if pose_kf.success:
                        scale = self.cfg.fixed_scale
                    else:
                        scale = self.cfg.fixed_scale

                T_rp = np.eye(4)
                T_rp[:3, :3] = pose_prev.R
                T_rp[:3,  3] = pose_prev.t.flatten() * scale
                
                # Bug 1: Correct accumulation formula.
                # T_rp from cv2.recoverPose is T_{cur <- ref}. Must invert to get T_{ref <- cur}.
                current_T_world_cam_candidate = compose_pose(self.T_world_cam, invert_pose(T_rp))
                tracking_successful_frame_to_frame = True
                self.last_T_rel = invert_pose(T_rp) # Store relative for constant velocity prediction

        # --- PnP Fallback if Frame-to-Frame Tracking Failed ---
        if not tracking_successful_frame_to_frame and kf is not None and len(kf.map_points) > 0:
            if len(match_kf) >= self.cfg.min_inliers:
                pts3d_pnp = []
                pts2d_pnp = []
                for i in range(len(match_kf)):
                    idx_ref = match_kf.idx_ref[i]
                    mp = kf.get_map_point_at_feature(idx_ref)
                    if mp is not None:
                        pts3d_pnp.append(mp.xyz)
                        pts2d_pnp.append(match_kf.pts_cur[i])

                if len(pts3d_pnp) >= 6: # Min 6 points for PnP
                    try:
                        ok, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                            np.array(pts3d_pnp, dtype=np.float32), 
                            np.array(pts2d_pnp, dtype=np.float32), 
                            self.camera.K, None, # No distortion coeffs
                            iterationsCount=100, 
                            reprojectionError=2.0, 
                            confidence=0.99,
                            flags=cv2.SOLVEPNP_EPNP
                        )

                        if ok and inliers_pnp is not None and len(inliers_pnp) >= self.cfg.min_inliers // 2:
                            R_cw, _ = cv2.Rodrigues(rvec)
                            t_cw = tvec.ravel()
                            
                            T_cam_world = np.eye(4)
                            T_cam_world[:3, :3] = R_cw
                            T_cam_world[:3, 3] = t_cw

                            current_T_world_cam_candidate = invert_pose(T_cam_world)
                            stats.num_inliers = len(inliers_pnp)
                            tracking_successful_pnp = True
                    except cv2.error:
                        pass # PnP can fail for various reasons"""

new_block = """        # --- Attempt PnP Tracking First (Primary Tracking) ---
        if kf is not None and len(kf.map_points) > 0 and len(match_kf) >= self.cfg.min_inliers:
            pts3d_pnp = []
            pts2d_pnp = []
            for i in range(len(match_kf)):
                idx_ref = match_kf.idx_ref[i]
                mp = kf.get_map_point_at_feature(idx_ref)
                if mp is not None:
                    pts3d_pnp.append(mp.xyz)
                    pts2d_pnp.append(match_kf.pts_cur[i])

            if len(pts3d_pnp) >= 10: # Require at least 10 points to trust PnP for tracking
                try:
                    ok, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                        np.array(pts3d_pnp, dtype=np.float32), 
                        np.array(pts2d_pnp, dtype=np.float32), 
                        self.camera.K, None, # No distortion coeffs
                        iterationsCount=300, 
                        reprojectionError=4.0, 
                        confidence=0.99,
                        flags=cv2.SOLVEPNP_EPNP
                    )

                    if ok and inliers_pnp is not None and len(inliers_pnp) >= max(10, self.cfg.min_inliers // 2):
                        R_cw, _ = cv2.Rodrigues(rvec)
                        t_cw = tvec.ravel()
                        
                        T_cam_world = np.eye(4)
                        T_cam_world[:3, :3] = R_cw
                        T_cam_world[:3, 3] = t_cw

                        current_T_world_cam_candidate = invert_pose(T_cam_world)
                        stats.num_inliers = len(inliers_pnp)
                        tracking_successful_pnp = True
                        
                        # Store relative for constant velocity prediction
                        self.last_T_rel = invert_pose(self.T_world_cam) @ current_T_world_cam_candidate
                except cv2.error:
                    pass # PnP can fail for various reasons

        # --- Fallback to Frame-to-Frame Tracking ---
        if not tracking_successful_pnp and len(match_prev) >= self.cfg.min_inliers:
            pose_prev = self.estimator.estimate(match_prev.pts_ref, match_prev.pts_cur)
            stats.num_inliers = pose_prev.num_inliers
            stats.H_score     = pose_prev.H_score 

            if pose_prev.success:
                # Calculate T_new using frame-to-frame logic (existing scale modes)
                scale = self.cfg.fixed_scale # Default scale
                if self.cfg.scale_mode == 'fixed':
                    scale = self.cfg.fixed_scale
                elif use_stereo or len(self.keyframes) >= 5: # Metric mode
                    if pose_kf.success:
                        # Bug 1 fix: Scale recovery expects T_{cur <- ref} 
                        # but estimator returns T_{ref <- cur}. Must invert.
                        T_cur_kf = invert_pose(pose_kf.transform_matrix_ref_from_cur())
                        if cur_depths is not None:
                            scale = self._recover_scale_stereo(match_kf, cur_depths, T_cur_kf[:3,:3], T_cur_kf[:3,3])
                        else:
                            scale = self._recover_scale(kf, match_kf, T_cur_kf[:3,:3], T_cur_kf[:3,3])
                    else:
                        scale = self._prev_scale
                else: # Monocular Bootstrap mode
                    if pose_kf.success:
                        scale = self.cfg.fixed_scale
                    else:
                        scale = self.cfg.fixed_scale

                T_rp = np.eye(4)
                T_rp[:3, :3] = pose_prev.R
                T_rp[:3,  3] = pose_prev.t.flatten() * scale
                
                # Bug 1: Correct accumulation formula.
                # T_rp from cv2.recoverPose is T_{cur <- ref}. Must invert to get T_{ref <- cur}.
                current_T_world_cam_candidate = compose_pose(self.T_world_cam, invert_pose(T_rp))
                tracking_successful_frame_to_frame = True
                self.last_T_rel = invert_pose(T_rp) # Store relative for constant velocity prediction"""

if old_block in content:
    content = content.replace(old_block, new_block)
    with open('vo_slam/pipeline.py', 'w') as f:
        f.write(content)
    print("Successfully patched tracking logic")
else:
    print("Could not find the exact old_block in pipeline.py. Exiting.")
