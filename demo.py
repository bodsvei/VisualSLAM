"""
demo.py
-------
Run the Visual Odometry pipeline.
Modes
-----
  1. KITTI dataset  – pass --kitti /path/to/sequence/image_0
  2. Video file     – pass --video  path/to/video.mp4
  3. Webcam         – pass --webcam
  4. Synthetic      – default (no arguments): renders a virtual flythrough scene

Usage examples
--------------
  python demo.py                                  # synthetic
  python demo.py --kitti /data/kitti/00/image_0  # KITTI
  python demo.py --video  input.mp4
  python demo.py --webcam

Press 'q' to quit,  's' to save a trajectory snapshot.
"""

import argparse
import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
import os
if os.environ.get("DISPLAY") or sys.platform == "darwin" or sys.platform == "win32":
    try:
        matplotlib.use("TkAgg")   # interactive — supports plt.show()
    except Exception:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")         # headless server — no display
import matplotlib.pyplot as plt

# ── Add parent directory so `vo` is importable when running directly ── #
sys.path.insert(0, str(Path(__file__).parent))

from vo_slam import (
    CameraModel, VOConfig, VisualOdometry, DetectorType,
    FeatureOverlay, TrajectoryPlot, plot_trajectory_static,
)


# ═══════════════════════════════════════════════════════════════════════ #
#  Synthetic scene generator                                              #
# ═══════════════════════════════════════════════════════════════════════ #

class SyntheticScene:
    """
    Generates a stream of synthetic camera frames by rendering a 3-D
    point cloud (random 3-D points) with a moving camera.
    Useful for quick testing without a real dataset.
    """

    def __init__(
        self,
        camera      : CameraModel,
        num_points  : int   = 600,
        scene_depth : float = 15.0,
        scene_width : float = 10.0,
        scene_height: float = 5.0,
        noise_std   : float = 0.5,   # pixel noise
    ):
        self.camera = camera
        rng   = np.random.RandomState(42)

        # 3-D points in world frame
        self._pts3d = np.column_stack([
            rng.uniform(-scene_width,  scene_width,  num_points),
            rng.uniform(-scene_height, scene_height, num_points),
            rng.uniform(2.0,           scene_depth,  num_points),
        ])
        self._noise_std = noise_std

        # Trajectory: straight forward with slight sine curve
        self._frame_id  = 0
        self._max_frames= 300

    def __iter__(self):
        return self

    def __next__(self):
        if self._frame_id >= self._max_frames:
            raise StopIteration

        t = self._frame_id * 0.05       # time parameter
        # Camera moves forward (−z in OpenCV convention) with lateral sine
        tx = 0.3 * np.sin(t * 0.5)
        ty = 0.0
        tz = -t                          # moving forward in world z

        # Simple Ry rotation around y axis
        angle = 0.02 * np.sin(t * 0.3)
        Ry    = np.array([
            [ np.cos(angle), 0, np.sin(angle)],
            [ 0,             1, 0            ],
            [-np.sin(angle), 0, np.cos(angle)],
        ])

        # T_world_cam  →  we need T_cam_world for projection
        R_cw = Ry.T
        t_cw = -Ry.T @ np.array([tx, ty, tz])

        img = self._render(R_cw, t_cw)
        self._frame_id += 1
        return img, self._frame_id - 1

    def _render(self, R_cw, t_cw):
        """Project 3-D points into the camera and render as a synthetic frame."""
        H, W = self.camera.height, self.camera.width
        img   = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (20, 20, 30)          # dark blue-ish background

        # Transform to camera frame
        pts_cam = (R_cw @ self._pts3d.T).T + t_cw   # (N, 3)
        in_front = pts_cam[:, 2] > 0.1

        # Project
        uv = self.camera.project(pts_cam[in_front])

        # Filter inside image
        in_img = (
            (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) &
            (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
        )
        uv_vis  = uv[in_img]
        z_vis   = pts_cam[in_front][in_img, 2]

        # Colour by depth
        z_norm = np.clip((z_vis - 1) / 20.0, 0, 1)

        noise = np.random.randn(*uv_vis.shape) * self._noise_std
        uv_vis = uv_vis + noise

        for (u, v), zn in zip(uv_vis.astype(int), z_norm):
            if 0 <= u < W and 0 <= v < H:
                b = int(50  + 80  * (1 - zn))
                g = int(180 + 60  * (1 - zn))
                r = int(80  + 100 * zn)
                cv2.circle(img, (u, v), 2, (b, g, r), -1)

        # Add a grid on the "ground"
        self._draw_grid(img, R_cw, t_cw)
        return img

    def _draw_grid(self, img, R_cw, t_cw):
        H, W = self.camera.height, self.camera.width
        K = self.camera.K
        for gx in range(-5, 6):
            for gz in range(0, 20):
                p3d = np.array([[gx * 2.0, -2.0, gz * 2.0]])
                pc  = (R_cw @ p3d.T).T + t_cw
                if pc[0, 2] < 0.1:
                    continue
                uv = self.camera.project(pc)
                if np.isnan(uv).any():
                    continue
                u, v = int(uv[0, 0]), int(uv[0, 1])
                if 0 <= u < W and 0 <= v < H:
                    cv2.circle(img, (u, v), 1, (60, 60, 60), -1)


# ═══════════════════════════════════════════════════════════════════════ #
#  Dataset loaders                                                        #
# ═══════════════════════════════════════════════════════════════════════ #

def kitti_loader(folder: str):
    """Yield (image, frame_id) from a KITTI image folder."""
    p = Path(folder)
    files = sorted(p.glob("*.png")) + sorted(p.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No images in {folder}")
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is not None:
            yield img, i


def video_loader(path: str):
    """Yield (frame, frame_id) from a video file."""
    cap = cv2.VideoCapture(path)
    fid = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame, fid
        fid += 1
    cap.release()


def webcam_loader(device: int = 0):
    """Yield (frame, frame_id) from webcam."""
    cap = cv2.VideoCapture(device)
    fid = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame, fid
        fid += 1
    cap.release()


# ═══════════════════════════════════════════════════════════════════════ #
#  Main runner                                                            #
# ═══════════════════════════════════════════════════════════════════════ #

def run(args):
    # ── Camera ─────────────────────────────────────────────────────── #
    if args.kitti:
        camera = CameraModel.kitti()
    elif args.webcam or args.video:
        cap = cv2.VideoCapture(0 if args.webcam else args.video)
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        camera = CameraModel.from_fov(60.0, W, H)
    else:
        camera = CameraModel.from_fov(70.0, width=1241, height=376)

    print(f"Camera: {camera}")

    # ── VO config ──────────────────────────────────────────────────── #
    cfg = VOConfig(
        detector_type    = DetectorType.ORB,
        max_features     = 1500,
        ratio_thresh     = 0.75,
        min_inliers      = 15,
        scale_mode       = "fixed",
        store_images     = False,
        kf_min_parallax  = 2.0,
        kf_min_frames    = 3,
        kf_max_frames    = 15,
    )
    vo = VisualOdometry(camera, cfg)

    # ── Source ─────────────────────────────────────────────────────── #
    if args.kitti:
        source = kitti_loader(args.kitti)
    elif args.video:
        source = video_loader(args.video)
    elif args.webcam:
        source = webcam_loader()
    else:
        print("No source specified – running synthetic scene demo.")
        scene  = SyntheticScene(camera)
        source = scene

    # ── Visualisation setup ────────────────────────────────────────── #
    traj_plotter = TrajectoryPlot(figsize=(500, 400))
    show_gui     = not args.no_gui

    # ── Standalone ORB tracker (mirrors orb.py) ────────────────────── #
    orb_detector = cv2.ORB_create(nfeatures=1000)
    orb_matcher  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    orb_prev_frame: np.ndarray | None = None
    orb_prev_kp                        = None
    orb_prev_des                       = None

    print("\nRunning VO. Press 'q' to quit, 's' to save trajectory.\n")

    for frame, fid in source:
        stats = vo.process(frame, timestamp=fid / 30.0)

        # ── ORB feature tracking display (like orb.py) ─────────────── #
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_kp, curr_des = orb_detector.detectAndCompute(curr_gray, None)

        if show_gui and orb_prev_des is not None and curr_des is not None \
                and len(orb_prev_des) > 0 and len(curr_des) > 0:
            matches = orb_matcher.knnMatch(orb_prev_des, curr_des, k=2)
            good_matches = [
                m for m_n in matches if len(m_n) == 2
                for m, n in [m_n] if m.distance < 0.75 * n.distance
            ]
            orb_display = cv2.drawMatches(
                orb_prev_frame, orb_prev_kp,
                frame,          curr_kp,
                good_matches,   None,
                matchColor      = (0, 255, 0),
                flags           = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            # Stamp match count so it's easy to read at a glance
            cv2.putText(
                orb_display,
                f"Matches: {len(good_matches)}  |  Frame {fid}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
            )
            cv2.imshow("ORB Tracking  (Left: prev  |  Right: curr)", orb_display)

        # Advance ORB state for next iteration
        orb_prev_frame = frame.copy()
        orb_prev_kp    = curr_kp
        orb_prev_des   = curr_des

        if show_gui:
            # Camera feed with HUD
            display = FeatureOverlay.draw_hud(
                frame,
                frame_id   = fid,
                num_kf     = len(vo.keyframes),
                num_mp     = len(vo.map_points),
                position   = vo.T_world_cam[:3, 3],
                is_kf      = stats.is_keyframe,
                process_ms = stats.process_ms,
            )
            cv2.imshow("Camera", display)

            # Trajectory map (rendered periodically to save time)
            if fid % 5 == 0 and len(vo.trajectory) > 1:
                traj_img = traj_plotter.render_2d(
                    vo.trajectory, vo.map_points, vo.keyframes
                )
                cv2.imshow("Trajectory", traj_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                _save_snapshot(vo, fid)

        # Console log every 10 frames
        if fid % 10 == 0:
            pos = vo.T_world_cam[:3, 3]
            kf_tag = " [KF]" if stats.is_keyframe else ""
            print(f"  Frame {fid:4d} | "
                  f"matched={stats.num_matched:4d} | "
                  f"inliers={stats.num_inliers:4d} | "
                  f"MPs={len(vo.map_points):5d} | "
                  f"pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) | "
                  f"{stats.process_ms:.1f}ms{kf_tag}")

    print("\n" + vo.summary())

    # ── Final trajectory plot ──────────────────────────────────────── #
    if len(vo.trajectory) > 1:
        out_path = args.output or "trajectory.png"
        fig = plot_trajectory_static(
            vo.trajectory, vo.map_points, vo.keyframes,
            save_path=out_path,
            title=f"VO Trajectory ({len(vo.trajectory)} frames, "
                  f"{len(vo.keyframes)} keyframes)",
        )
        print(f"\nTrajectory saved to: {out_path}")
        if show_gui and matplotlib.get_backend() != "Agg":
            plt.show()
        else:
            print(f"Trajectory saved to: {out_path}")
        plt.close(fig)

    if show_gui:
        cv2.destroyAllWindows()


def _save_snapshot(vo, fid):
    path = f"snapshot_frame{fid:04d}.png"
    if len(vo.trajectory) > 1:
        fig = plot_trajectory_static(
            vo.trajectory, vo.map_points, vo.keyframes,
            save_path=path,
            title=f"Snapshot @ frame {fid}",
        )
        plt.close(fig)
        print(f"  → Snapshot saved: {path}")


# ═══════════════════════════════════════════════════════════════════════ #
#  CLI                                                                    #
# ═══════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Odometry Demo")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--kitti",  type=str, help="Path to KITTI image folder")
    src.add_argument("--video",  type=str, help="Path to video file")
    src.add_argument("--webcam", action="store_true", help="Use webcam (device 0)")
    parser.add_argument("--output",  type=str, default="trajectory.png",
                        help="Output trajectory image path")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable OpenCV windows (headless mode)")
    args = parser.parse_args()
    run(args)