import re

with open('vo_slam/pipeline.py', 'r') as f:
    code = f.read()

# Cleaning up the nested double-else from previous failed edit
new_track_body = """        # ── 3. Tracking Logic Choice ────────────────────────────────── #
        if self.cfg.scale_mode == 'fixed' or len(self.keyframes) < 5:
            # Bootstrap or Fixed mode
            scale = self.cfg.fixed_scale
            if self.cfg.scale_mode == 'fixed':
                # Pure frame-to-frame (smooth)
                T_rp = np.eye(4)
                T_rp[:3, :3] = pose_prev.R
                T_rp[:3,  3] = pose_prev.t.flatten() * scale
                T_new = compose_pose(self.T_world_cam, invert_pose(T_rp))
            else:
                # KF-relative bootstrap
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
                T_new = compose_pose(self.T_world_cam, invert_pose(T_rp))"""

pattern = r'# ── 3\. Tracking Logic Choice ──.*?invert_pose\(T_rp\)\)'
code = re.sub(pattern, new_track_body, code, flags=re.DOTALL)

with open('vo_slam/pipeline.py', 'w') as f:
    f.write(code)
