import numpy as np
import g2o

# World frame
T1 = np.eye(4); T1[0,3] = 1.0 # Cam 1 at X=1
T2 = np.eye(4); T2[0,3] = 2.0 # Cam 2 at X=2

X1 = g2o.SE3Quat(np.eye(3), [-1., 0, 0]) # T_cam1_world
X2 = g2o.SE3Quat(np.eye(3), [-2., 0, 0]) # T_cam2_world

# g2o uses X1.inverse() * X2
err = X1.inverse() * X2
print("inv(X1)*X2 translation:", err.translation())
