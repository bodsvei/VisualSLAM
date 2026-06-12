import re

with open('vo_slam/pipeline.py', 'r') as f:
    code = f.read()

# Scale mode 'median_depth' often needs a good initial scale to avoid feedback loop collapse.
# Let's force the first few keyframes to use fixed_scale to bootstrap.

boot_code = """        # ── Scale recovery ─────────────────────────────────────────── #
        if self.cfg.scale_mode == 'fixed' or len(self.keyframes) < 5:
            # Bootstrap with fixed scale for the first 5 KFs
            scale = self.cfg.fixed_scale
            if self.cfg.scale_mode == 'fixed':
                # Differential tracking
                T_rp = np.eye(4)
                T_rp[:3, :3] = pose_prev.R
                T_rp[:3,  3] = pose_prev.t.flatten() * scale
                T_new = compose_pose(self.T_world_cam, invert_pose(T_rp))
            else:
                # KF-relative tracking (bootstrapping)
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
        else:"""

code = code.replace("        if self.cfg.scale_mode == 'fixed':", boot_code)

with open('vo_slam/pipeline.py', 'w') as f:
    f.write(code)

