import cv2
import numpy as np

# Camera 1 at origin
# Camera 2 at Z=1 (moved forward 1m)
# p2 = p1 - [0, 0, 1]

K = np.eye(3)
pts3d = np.array([[0,0,10], [1,0,10], [0,1,10], [-1,0,10], [0,-1,10], [5,5,10], [2,-2,15], [-3,4,12]], dtype=np.float32)
pts1 = (pts3d[:, :2] / pts3d[:, 2:])

pts3d_2 = pts3d - [0, 0, 1]
pts2 = (pts3d_2[:, :2] / pts3d_2[:, 2:])

E, _ = cv2.findEssentialMat(pts1, pts2, K)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

print("Recovered t:", t.ravel())

T_rp = np.eye(4); T_rp[:3,:3]=R; T_rp[:3,3]=t.ravel()
T_fwd = np.linalg.inv(T_rp)
print("Forward t (inverted):", T_fwd[:3, 3])

