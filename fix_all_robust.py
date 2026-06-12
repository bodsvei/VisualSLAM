import re

with open('vo_slam/pipeline.py', 'r') as f:
    code = f.read()

# 1. Implement IQR Robust Scale Recovery
robust_scale_code = """    def _recover_scale(self, kf: Keyframe, match_kf, R: np.ndarray, t: np.ndarray) -> float:
        \"\"\"
        Estimate relative scale between last KF and current frame.
        Uses IQR filtering to reject depth outliers for a robust median estimate.
        \"\"\"
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

        if len(mps_ref) < 8:
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

        # ── Robust Ratio Estimation ─────────────────────────────────── #
        valid = (depths_cur_unscaled > 0.1) & (np.array(depths_ref) > 0.1)
        if np.sum(valid) < 5:
            return self.cfg.fixed_scale

        ratios = np.array(depths_ref)[valid] / depths_cur_unscaled[valid]
        
        # IQR filtering for robustness
        q1, q3 = np.percentile(ratios, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered_ratios = ratios[(ratios >= lower) & (ratios <= upper)]
        
        if len(filtered_ratios) < 3:
            ratio = np.median(ratios)
        else:
            ratio = np.median(filtered_ratios)

        return float(np.clip(ratio, self.cfg.scale_clamp_min, self.cfg.scale_clamp_max))"""

code = re.sub(r'    def _recover_scale\(self, kf: Keyframe, match_kf, R: np.ndarray, t: np.ndarray\) -> float:.*?return float\(np.clip\(ratio, self.cfg.scale_clamp_min, self.cfg.scale_clamp_max\)\)', robust_scale_code, code, flags=re.DOTALL)

with open('vo_slam/pipeline.py', 'w') as f:
    f.write(code)

# 2. Relax Loop Detection and BA frequency
with open('vo_slam/loop_detector.py', 'r') as f:
    ld_code = f.read()
ld_code = ld_code.replace('min_loop_gap_frames: int   = 200', 'min_loop_gap_frames: int   = 100')
ld_code = ld_code.replace('min_bow_score       : float = 0.012', 'min_bow_score       : float = 0.009')
with open('vo_slam/loop_detector.py', 'w') as f:
    f.write(ld_code)

with open('vo_slam/local_mapping.py', 'r') as f:
    lm_code = f.read()
# Make covisibility graph less strict to allow more BA runs
lm_code = lm_code.replace('CovisibilityGraph()', 'CovisibilityGraph(min_shared=10)')
with open('vo_slam/local_mapping.py', 'w') as f:
    f.write(lm_code)

