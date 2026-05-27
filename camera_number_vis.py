import numpy as np

data = np.load("calibration.npz")

# ['K', 'dist', 'rvecs', 'tvecs', 'rms', 'mean_reprojection_error']

print(data.files)
print(data['K'])
print(data['dist'])
print(data['rvecs'])
print(data['tvecs'])
print(data['rms'])
print(data['mean_reprojection_error'])