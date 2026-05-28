import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 2)) # VERY wide figure
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

pos_x = [0, 10, 20, 30]
pos_z = [0, 10, 20, 30]
dir_x = [0.707, 0.707, 0.707, 0.707] # 45 degree angle
dir_z = [0.707, 0.707, 0.707, 0.707]

# uv angles (should look horribly distorted towards horizontal)
ax.quiver(pos_x, pos_z, dir_x, dir_z, color="red",
          scale=5, angles='uv', zorder=4)

# xy angles (should point exactly 45 degrees, matching the trajectory line)
ax.quiver(pos_x, pos_z, dir_x, dir_z, color="white",
          scale=5, angles='xy', zorder=5)

ax.set_aspect('equal')
fig.savefig("test_quiver_3.png")
