import cv2
import numpy as np

# Let's set up a clear 3D scenario.
# Camera 1 at origin looking along Z.
# Camera 2 at X=10, looking along Z.
# So Cam 2 is moved to the RIGHT.
# A point at X=0, Z=20 in world is at X=0, Z=20 in Cam 1.
# In Cam 2 (at X=10), the point is at X=-10, Z=20.
# So p2 = p1 - [10, 0, 0]^T
# So p2 = R p1 + t, where R = I, t = [-10, 0, 0]^T.

# Let's see what recoverPose gives us!

pts3d = np.array([
    [0, 0, 20],
    [5, 0, 20],
    [-5, 0, 20],
    [0, 5, 20],
    [0, -5, 20],
    [2, 2, 15],
    [-2, -2, 15]
], dtype=np.float32)

# Camera 1
K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float32)

def project(pts, R, t):
    p_cam = (R @ pts.T).T + t
    p_img = p_cam[:, :2] / p_cam[:, 2:]
    return (K[:2,:2] @ p_img.T).T + K[:2, 2]

# Cam 1: R=I, t=0
pts1 = project(pts3d, np.eye(3), np.zeros(3))

# Cam 2: at world X=10. p_cam = p_world - [10,0,0]
t2 = np.array([-10, 0, 0])
pts2 = project(pts3d, np.eye(3), t2)

# Recover pose
E, _ = cv2.findEssentialMat(pts1, pts2, K)
_, R_rec, t_rec, _ = cv2.recoverPose(E, pts1, pts2, K)

print("Recovered t (should be normalized):", t_rec.ravel())
print("Expected t direction if p2 = R p1 + t:", t2 / np.linalg.norm(t2))
print("Expected t direction if p1 = R p2 + t:", -t2 / np.linalg.norm(t2))

