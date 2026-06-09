import numpy as np
from scipy.spatial.transform import Rotation as R

def get_euler(T):
    return R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)

# Try a flip
R_flip = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
print("Flip:", R.from_matrix(R_flip).as_euler('xyz', degrees=True))
