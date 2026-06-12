import numpy as np
from vo_slam.motion import compose_pose, invert_pose

# Simulating the exact tracking logic over 15 frames of a 90 deg turn.
# Suppose the car moves on a quarter circle. Radius R = 100 / (pi/2) = 63.66m.
# Arc length = 100m. Speed = 1m/frame. N = 100 frames.
# Frame delta between KFs = 3.

world_poses = [np.eye(4)]

for i in range(1, 4):
    # True motion from KF (frame 0) to frame i
    angle = i * (np.pi / 2 / 100)  # 90 deg over 100 frames
    
    # Ground truth position in world
    pos_x = 63.66 * (1 - np.cos(angle))
    pos_z = 63.66 * np.sin(angle)
    
    T_w_c = np.eye(4)
    T_w_c[0, 3] = pos_x
    T_w_c[2, 3] = pos_z
    T_w_c[:3, :3] = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    
    # KF is at origin (I).
    # recoverPose returns T_cur_kf such that X_cur = R_cur_kf X_kf + t_cur_kf
    # T_cur_kf = invert(T_w_c)
    T_c_kf = invert_pose(T_w_c)
    
    R = T_c_kf[:3, :3]
    t = T_c_kf[:3, 3]
    
    # Scale t to unit length (simulate what recoverPose does)
    t_unit = t / np.linalg.norm(t)
    
    # Now simulate pipeline.py logic:
    # scale = 1.0 * (frame_id - kf_id) = 1.0 * i
    scale = i * 1.0
    
    T_rp = np.eye(4)
    T_rp[:3, :3] = R
    T_rp[:3, 3] = t_unit * scale
    
    # T_new = kf.T_world_cam @ invert_pose(T_rp)
    T_new = compose_pose(np.eye(4), invert_pose(T_rp))
    
    print(f"Frame {i}:")
    print(f"  GT  pos: {pos_x:.3f}, {pos_z:.3f}")
    print(f"  Est pos: {T_new[0,3]:.3f}, {T_new[2,3]:.3f}")

