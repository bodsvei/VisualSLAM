import re

with open('vo_slam/pipeline.py', 'r') as f:
    code = f.read()

# 1. Fix _track accumulation: remove invert_pose
# Fixed scale path
code = code.replace(
    'T_new = compose_pose(self.T_world_cam, invert_pose(T_rp))',
    'T_new = compose_pose(self.T_world_cam, T_rp)'
)
# Median depth path (KF-relative)
code = code.replace(
    'T_new = compose_pose(kf.T_world_cam, invert_pose(T_rp))',
    'T_new = compose_pose(kf.T_world_cam, T_rp)'
)

# 2. Fix _recover_scale projection: P2 must be K @ T_{cur <- ref}
code = code.replace(
    'P2 = self.camera.K @ T_rel[:3]',
    'P2 = self.camera.K @ invert_pose(T_rel)[:3]'
)

with open('vo_slam/pipeline.py', 'w') as f:
    f.write(code)

