import cv2
import numpy as np
from pathlib import Path
from vo_slam import CameraModel, VisualOdometry, VOConfig, DetectorType

# ── Config ────────────────────────────────────────────────────────── #
SEQ        = "00"
KITTI_ROOT = Path("kitti")          # ← change this
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

if not CALIB_FILE.exists() or not GT_FILE.exists():
    print(f"KITTI dataset files not found at {KITTI_ROOT}. Please download and set KITTI_ROOT.")
    exit(1)

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

for i, img_path in enumerate(images):
    img   = cv2.imread(str(img_path))
    stats = vo.process(img, timestamp=i / 10.0)

    if i % 50 == 0:
        pos = vo.T_world_cam[:3, 3]
        gt_pos = gt[i][:3, 3]
        print(f"Frame {i:4d} | est=({pos[0]:7.2f},{pos[2]:7.2f}) | "
              f"gt=({gt_pos[0]:7.2f},{gt_pos[2]:7.2f}) | "
              f"KFs={len(vo.keyframes):3d} | MPs={len(vo.map_points):5d}")

# ── Save estimated poses in KITTI format ─────────────────────────── #
out_path = f"results_{SEQ}.txt"
with open(out_path, "w") as f:
    for T in vo.pose_graph.poses:
        row = T[:3].flatten()
        f.write(" ".join(f"{v:.6e}" for v in row) + "\n")

print(f"\nPoses saved to {out_path}")
print(vo.summary())
