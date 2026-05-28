import cv2
import numpy as np
from pathlib import Path
from vo_slam import CameraModel, VisualOdometry, VOConfig, DetectorType

# ── Config ────────────────────────────────────────────────────────── #
SEQ = "00"

KITTI_ROOT = Path("/Users/anirudhsinghair/Documents/GitHub/VisualSLAM/KITTI")

IMG_DIR    = KITTI_ROOT / "sequences" / SEQ / "image_0"
CALIB_FILE = KITTI_ROOT / "sequences" / SEQ / "calib.txt"
GT_FILE    = KITTI_ROOT / "poses" / f"{SEQ}.txt"

# ── Load calibration ──────────────────────────────────────────────── #
def load_calib(path):
    with open(path) as f:
        for line in f:
            if line.startswith("P0:"):
                vals = list(map(float, line.strip().split()[1:]))
                P = np.array(vals).reshape(3, 4)
                return P[:3, :3]

def load_poses(path):
    poses = []
    with open(path) as f:
        for line in f:
            T = np.array(list(map(float, line.strip().split()))).reshape(3,4)
            poses.append(np.vstack([T, [0,0,0,1]]))
    return poses

K  = load_calib(CALIB_FILE)
gt = load_poses(GT_FILE)

camera = CameraModel.from_matrix(K, width=1241, height=376)

# ── VO ────────────────────────────────────────────────────────────── #
cfg = VOConfig(
    detector_type = DetectorType.ORB,
    max_features  = 2000,
    min_inliers   = 20,
    scale_mode    = "median_depth",
)
vo = VisualOdometry(camera, cfg)

# ── Run ───────────────────────────────────────────────────────────── #
images = sorted(IMG_DIR.glob("*.png"))
print(f"Sequence {SEQ}: {len(images)} frames, {len(gt)} GT poses\n")

last_good_T = np.eye(4)   # fallback pose

all_poses = []             # collect one pose per frame

for i, img_path in enumerate(images):
    img   = cv2.imread(str(img_path))
    stats = vo.process(img, timestamp=i / 10.0)

    if np.isfinite(vo.T_world_cam).all():
        last_good_T = vo.T_world_cam.copy()

    all_poses.append(last_good_T.copy())   # always append, even on LOST

    if i % 50 == 0:
        pos    = last_good_T[:3, 3]
        gt_pos = gt[i][:3, 3]
        state  = vo.state.name
        print(f"Frame {i:4d} | est=({pos[0]:7.2f},{pos[2]:7.2f}) | "
              f"gt=({gt_pos[0]:7.2f},{gt_pos[2]:7.2f}) | "
              f"KFs={len(vo.keyframes):3d} | MPs={len(vo.map_points):5d} | "
              f"{state}")

# ── Save — one line per frame ──────────────────────────────────────── #
out_path = f"results_{SEQ}.txt"
with open(out_path, "w") as f:
    for T in all_poses:
        row = T[:3].flatten()             # 3×4 → 12 values
        f.write(" ".join(f"{v:.6e}" for v in row) + "\n")

print(f"\nSaved {len(all_poses)} poses → {out_path}")
assert len(all_poses) == len(images), "Pose count mismatch!"
print(vo.summary())