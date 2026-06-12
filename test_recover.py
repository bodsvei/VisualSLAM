import cv2
import numpy as np

# Camera 1 at origin
# Camera 2 translated by (1, 0, 0) and rotated 90 deg around Y
# X_world = X_1
# X_2 = R_{21} X_1 + t_{21}
# Let's say cam 2 is looking left (turned left by 90 deg). 
# World point at (0, 0, 5) -> Cam 1: (0, 0, 5).
# Cam 2 is at (1, 0, 0) in world, facing (-1, 0, 0).
# So in Cam 2, the point (0, 0, 5) should be at Z=1, X=5.
# Let's just create 5 random 3D points and project them.

P1 = np.array([[1,0,0,0],
               [0,1,0,0],
               [0,0,1,0]], dtype=float)

# 90 deg around Y
R = np.array([[0, 0, -1],
              [0, 1, 0],
              [1, 0, 0]], dtype=float)
t = np.array([[0], [0], [1]], dtype=float)

P2 = np.hstack((R, t))

K = np.array([[100, 0, 50],
              [0, 100, 50],
              [0,   0,  1]], dtype=float)

pts3d = np.random.rand(10, 3) * 5 + np.array([0, 0, 10])
pts3d = pts3d.astype(np.float32)

pts1 = []
pts2 = []
for p in pts3d:
    p1 = K @ P1 @ np.append(p, 1)
    pts1.append((p1[:2] / p1[2]))
    
    p2 = K @ P2 @ np.append(p, 1)
    pts2.append((p2[:2] / p2[2]))

pts1 = np.array(pts1)
pts2 = np.array(pts2)

E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R_rec, t_rec, mask = cv2.recoverPose(E, pts1, pts2, K)

print("Original R:")
print(R)
print("Recovered R:")
print(R_rec)

print("Original t (normalized):")
print(t / np.linalg.norm(t))
print("Recovered t:")
print(t_rec)

