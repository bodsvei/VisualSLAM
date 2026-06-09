import cv2
import numpy as np

K = np.eye(3)

# Camera 1 at origin, Camera 2 at z=1 (x1 = x2 + [0,0,1] => x1 = x2 + t => t=[0,0,1])
# If x1 = R x2 + t, then t = [0, 0, 1]
# If x2 = R x1 + t, then t = [0, 0, -1]

# Let's generate points
np.random.seed(42)
pts_w = np.random.uniform(-5, 5, (100, 3)).astype(np.float64)
pts_w[:, 2] += 20  # points in front of both cameras

pts1 = pts_w[:, :2] / pts_w[:, 2:3]

# T_w_1 = I
# T_w_2: moved 1 unit along +X and rotated 10 deg around Y
angle = np.radians(10)
R_true = np.array([
    [np.cos(angle), 0, np.sin(angle)],
    [0, 1, 0],
    [-np.sin(angle), 0, np.cos(angle)]
])
t_true = np.array([1.0, 0.0, 0.0]) # T_w_2 translation

# x_w = R_w_2 * x_2 + t_w_2 => x_2 = R_w_2^T * (x_w - t_w_2)
pts2_3d = (pts_w - t_true) @ R_true
pts2 = pts2_3d[:, :2] / pts2_3d[:, 2:3]

E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R_est, t_est, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

print("R_est:\n", np.round(R_est, 3))
print("t_est:\n", np.round(t_est.ravel(), 3))

print("R_true (world->cam2):\n", np.round(R_true.T, 3))
print("R_true (cam2->world):\n", np.round(R_true, 3))

print("If x1 = R*x2 + t:")
t1 = t_true / np.linalg.norm(t_true)
print("t should be:", np.round(t1, 3))

print("If x2 = R*x1 + t:")
t2 = -R_true.T @ t_true
t2 = t2 / np.linalg.norm(t2)
print("t should be:", np.round(t2, 3))
