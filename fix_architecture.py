import re

with open('vo_slam/pipeline.py', 'r') as f:
    code = f.read()

# 1. Update _track to Reuse MapPoints and build Covisibility Graph
# Finding the 'if do_kf:' block
kf_logic_pattern = r'if do_kf:(.*?)new_kf = Keyframe\('
kf_logic_replacement = """if do_kf:
            # ── 1. MapPoint Reuse (Covisibility) ────────────────── #
            # Link current features to existing MapPoints from last KF
            matched_mps = []
            if match_kf is not None:
                for i in range(len(match_kf)):
                    idx_ref = match_kf.idx_ref[i]
                    idx_cur = match_kf.idx_cur[i]
                    mp = kf.get_map_point_at_feature(idx_ref)
                    if mp is not None:
                        # Record observation in existing MapPoint
                        mp.obs[self.kf_id] = int(idx_cur)
                        matched_mps.append(mp)

            # ── 2. New Triangulation ────────────────────────────── #
            new_mps = []
            if pose_kf is not None and pose_kf.success:
                inlier_mask        = pose_kf.inlier_mask
                # Only triangulate features that DON'T already have a MapPoint
                # (to avoid redundant points and keep map sparse)
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
            
            """

code = re.sub(kf_logic_pattern, kf_logic_replacement, code, flags=re.DOTALL)

# Update Keyframe instantiation to use all_kf_mps
code = code.replace('map_points  = new_mps,', 'map_points  = all_kf_mps,')

# 2. Fix Keyframe Decision to use accumulated motion
kf_decision_pattern = r'# ── Keyframe Decision ────────────────────────────────────── #.*?do_kf, kf_reason = self\.kf_selector\.should_insert\(.*?\)'
kf_decision_replacement = """        # ── 5. Keyframe Decision (Accumulated Motion) ──────────────── #
        # Relative motion from last KF to current
        T_kf_cur = compose_pose(invert_pose(kf.T_world_cam), self.T_world_cam)
        R_rel_kf = T_kf_cur[:3, :3]
        
        do_kf, kf_reason = self.kf_selector.should_insert(
            last_kf     = kf,
            R_rel       = R_rel_kf,
            pts_ref     = match_kf.pts_ref,
            pts_cur     = match_kf.pts_cur,
            num_tracked = len(match_kf),
        )"""

code = re.sub(kf_decision_pattern, kf_decision_replacement, code, flags=re.DOTALL)

with open('vo_slam/pipeline.py', 'w') as f:
    f.write(code)

