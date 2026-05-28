import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# A straight line then a turn
pos_x = [0, 0, 0, 0, 10, 20, 30]
pos_z = [0, 10, 20, 30, 40, 50, 50]

# Initially pointing along +Z (0, 1)
# Then turns to point along +X (1, 0)
dir_x = [0, 0, 0, 0.707, 1, 1, 1]
dir_z = [1, 1, 1, 0.707, 0, 0, 0]

ax.plot(pos_x, pos_z, color="red", linewidth=2)

# Version 1: default angles='uv' and scale=30
ax.quiver(pos_x, pos_z, dir_x, dir_z, color="white",
          scale=30, width=0.005, headwidth=4, headlength=4, 
          pivot='tail', zorder=4)

ax.set_aspect('equal')
fig.savefig("test_quiver_1.png")
plt.close(fig)

# Version 2: angles='xy', scale_units='xy', scale=0.1 (so arrow is 10 data units long)
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.plot(pos_x, pos_z, color="red", linewidth=2)
ax.quiver(pos_x, pos_z, dir_x, dir_z, color="white", angles='xy', scale_units='xy',
          scale=0.1, width=0.01, headwidth=4, headlength=4, 
          pivot='tail', zorder=4)

ax.set_aspect('equal')
fig.savefig("test_quiver_2.png")
