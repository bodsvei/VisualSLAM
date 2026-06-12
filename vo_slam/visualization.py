"""
visualization.py
----------------
Visualisation utilities for Visual Odometry.

Provides
--------
  FeatureOverlay  – draw keypoints, matches, optical-flow tracks on a frame
  TrajectoryPlot  – live Matplotlib 2-D/3-D trajectory + point cloud
  VOVisualizer    – composite display (OpenCV window + trajectory panel)
"""

from __future__ import annotations
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe in all envs)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
from typing import List, Optional, Tuple

from .features import FrameFeatures, MatchResult
from .triangulation import MapPoint
from .keyframe import Keyframe


# ═══════════════════════════════════════════════════════════════════════ #
#  Colour palette                                                         #
# ═══════════════════════════════════════════════════════════════════════ #

_GREEN  = (0, 220,   0)
_BLUE   = (255,  80,  10)
_RED    = (0,   30, 230)
_YELLOW = (0,  230, 230)
_WHITE  = (255, 255, 255)
_GRAY   = (140, 140, 140)


# ═══════════════════════════════════════════════════════════════════════ #
#  Feature overlay (on raw frame)                                         #
# ═══════════════════════════════════════════════════════════════════════ #

class FeatureOverlay:
    """Draw feature information onto a copy of the frame."""

    @staticmethod
    def draw_keypoints(
        img   : np.ndarray,
        feats : FrameFeatures,
        color : Tuple[int, int, int] = _GREEN,
        radius: int = 3,
    ) -> np.ndarray:
        out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for kp in feats.keypoints:
            cv2.circle(out, (int(kp.pt[0]), int(kp.pt[1])), radius, color, -1)
        return out

    @staticmethod
    def draw_matches(
        img_ref  : np.ndarray,
        img_cur  : np.ndarray,
        matches  : MatchResult,
        max_draw : int = 200,
    ) -> np.ndarray:
        """Side-by-side match visualisation."""
        h1, w1 = img_ref.shape[:2]
        h2, w2 = img_cur.shape[:2]
        out_h   = max(h1, h2)
        out_w   = w1 + w2
        canvas  = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        ref_bgr = _to_bgr(img_ref)
        cur_bgr = _to_bgr(img_cur)
        canvas[:h1, :w1]    = ref_bgr
        canvas[:h2, w1:w1+w2] = cur_bgr

        indices = np.random.choice(
            len(matches), min(max_draw, len(matches)), replace=False
        ) if len(matches) > 0 else []

        for i in indices:
            pt1 = (int(matches.pts_ref[i, 0]), int(matches.pts_ref[i, 1]))
            pt2 = (int(matches.pts_cur[i, 0]) + w1, int(matches.pts_cur[i, 1]))
            col = _random_color(i)
            cv2.line(canvas, pt1, pt2, col, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, col, -1)
            cv2.circle(canvas, pt2, 3, col, -1)

        cv2.putText(canvas, f"Matches: {len(matches)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def draw_optical_flow(
        img     : np.ndarray,
        pts_ref : np.ndarray,
        pts_cur : np.ndarray,
        max_draw: int = 300,
    ) -> np.ndarray:
        """Draw LK flow vectors on the current frame."""
        out = _to_bgr(img).copy()
        N   = min(max_draw, len(pts_ref))
        for i in range(N):
            p1 = tuple(pts_ref[i].astype(int))
            p2 = tuple(pts_cur[i].astype(int))
            cv2.line(out, p1, p2, _GREEN, 1, cv2.LINE_AA)
            cv2.circle(out, p2, 2, _RED, -1)
        return out

    @staticmethod
    def draw_inlier_outliers(
        img     : np.ndarray,
        pts     : np.ndarray,
        mask    : np.ndarray,
    ) -> np.ndarray:
        """Colour inlier (green) and outlier (red) points."""
        out = _to_bgr(img).copy()
        for i, pt in enumerate(pts):
            color = _GREEN if mask[i] else _RED
            cv2.circle(out, (int(pt[0]), int(pt[1])), 3, color, -1)
        n_in  = int(mask.sum())
        n_out = len(mask) - n_in
        cv2.putText(out, f"Inliers: {n_in}  Outliers: {n_out}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 1, cv2.LINE_AA)
        return out

    @staticmethod
    def draw_hud(
        img        : np.ndarray,
        frame_id   : int,
        num_kf     : int,
        num_mp     : int,
        position   : np.ndarray,
        is_kf      : bool = False,
        process_ms : float = 0.0,
    ) -> np.ndarray:
        """Heads-up display overlay."""
        out   = _to_bgr(img).copy()
        lines = [
            f"Frame   : {frame_id}",
            f"KF/MP   : {num_kf} / {num_mp}",
            f"Pos X,Z : {position[0]:.2f}, {position[2]:.2f}",
            f"Time    : {process_ms:.1f} ms",
        ]
        y = 25
        for line in lines:
            cv2.putText(out, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA)
            y += 22

        if is_kf:
            h, w = out.shape[:2]
            cv2.rectangle(out, (w-120, 5), (w-5, 30), _YELLOW, -1)
            cv2.putText(out, "KEYFRAME", (w-115, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return out


# ═══════════════════════════════════════════════════════════════════════ #
#  Trajectory plot (Matplotlib → numpy image)                             #
# ═══════════════════════════════════════════════════════════════════════ #

class TrajectoryPlot:
    """
    Renders trajectory and point cloud to a numpy image via Matplotlib.
    Call `render(trajectory, map_points, keyframes)` each frame.

    Returns an (H, W, 3) BGR numpy array suitable for cv2.imshow.
    """

    def __init__(
        self,
        figsize    : Tuple[int, int] = (1200, 1200),   # hi-res live panel
        point_size : float = 0.8,
        traj_color : str   = "red",
        kf_color   : str   = "#4A90D9",                # blue keyframe dots
    ):
        self.fig_w, self.fig_h = figsize
        self.point_size = point_size
        self.traj_color = traj_color
        self.kf_color   = kf_color

    def render_2d(
        self,
        trajectory : np.ndarray,          # (N, 3) XYZ
        map_points : List[MapPoint] = [],
        keyframes  : List[Keyframe] = [],
        axes       : str            = "xz",   # which axes to plot
    ) -> np.ndarray:
        """Top-down 2-D trajectory plot. Returns BGR image."""
        fig, ax = plt.subplots(figsize=(self.fig_w/150, self.fig_h/150), dpi=150)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        ix, iy = _axis_indices(axes)

        # Map points
        if map_points:
            mp_xyz = np.array([mp.xyz for mp in map_points])
            ax.scatter(mp_xyz[:, ix], mp_xyz[:, iy],
                       s=self.point_size, c="yellow", alpha=0.5, linewidths=0)

        # Trajectory
        if len(trajectory) > 1:
            ax.plot(trajectory[:, ix], trajectory[:, iy],
                    color=self.traj_color, linewidth=3.0, alpha=1.0, label="Trajectory")
            # Current position
            ax.scatter(trajectory[-1, ix], trajectory[-1, iy],
                       c="white", s=60, zorder=5)
            # Start position
            ax.scatter(trajectory[0, ix], trajectory[0, iy],
                       c='white', marker='s', s=40, zorder=6, label="Start")

        # Keyframe positions — blue dots, no per-KF arrows
        if keyframes:
            pos_x = [kf.T_world_cam[ix, 3] for kf in keyframes]
            pos_y = [kf.T_world_cam[iy, 3] for kf in keyframes]
            ax.scatter(pos_x, pos_y, s=25, c=self.kf_color,
                       zorder=4, label="Keyframes", linewidths=0)

            # Single camera-direction arrow on the LATEST keyframe only.
            # Extract the forward (+Z in camera frame) direction from the
            # rotation matrix and project it onto the chosen 2-D axes.
            latest = keyframes[-1]
            R = latest.T_world_cam[:3, :3]   # world ← cam rotation
            fwd_cam   = np.array([0.0, 0.0, 1.0])
            fwd_world = R @ fwd_cam           # forward in world frame
            dx = fwd_world[ix]
            dy = fwd_world[iy]
            length = max(np.hypot(dx, dy), 1e-6)
            dx /= length
            dy /= length

            # Scale arrow to ~3% of the current axis span for visibility
            xlim = ax.get_xlim() if ax.get_xlim() != (0.0, 1.0) else (-10, 10)
            span = max(abs(xlim[1] - xlim[0]), 1.0)
            arrow_len = span * 0.04

            ax.annotate(
                "", 
                xy=(pos_x[-1] + dx * arrow_len, pos_y[-1] + dy * arrow_len),
                xytext=(pos_x[-1], pos_y[-1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#FFD700",      # gold — stands out against blue dots + red traj
                    lw=2.0,
                    mutation_scale=14,
                ),
                zorder=7,
            )

        ax.set_xlabel(axes[0].upper(), color="white", weight='bold')
        ax.set_ylabel(axes[1].upper(), color="white", weight='bold')
        ax.tick_params(colors="white")
        ax.spines[:].set_color("white")
        ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')
        
        ax.set_title("VO Trajectory", color="white", fontsize=11, weight='bold')
        ax.legend(fontsize=8, facecolor="black", labelcolor="white", loc='lower right', edgecolor="white")
        fig.tight_layout(pad=0.5)

        img = _fig_to_bgr(fig)
        plt.close(fig)
        return img

    def render_3d(
        self,
        trajectory : np.ndarray,
        map_points : List[MapPoint] = [],
        keyframes  : List[Keyframe] = [],
    ) -> np.ndarray:
        """3-D trajectory plot. Returns BGR image."""
        fig = plt.figure(figsize=(self.fig_w/100, self.fig_h/100), dpi=100)
        fig.patch.set_facecolor("black")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("black")

        if map_points:
            mp_xyz = np.array([mp.xyz for mp in map_points])
            ax.scatter(mp_xyz[:, 0], mp_xyz[:, 1], mp_xyz[:, 2],
                       s=self.point_size, c="yellow", alpha=0.3)

        if len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                    color="red", linewidth=2.0)
            ax.scatter(*trajectory[-1], c="white", s=60)

        if keyframes:
            kf_pos = np.array([kf.position for kf in keyframes])
            ax.scatter(kf_pos[:, 0], kf_pos[:, 1], kf_pos[:, 2],
                       c="white", s=25)

        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.set_zlabel("Z", color="white")
        ax.set_title("VO Trajectory 3D", color="white")

        ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

        fig.tight_layout()
        img = _fig_to_bgr(fig)
        plt.close(fig)
        return img


# ═══════════════════════════════════════════════════════════════════════ #
#  Composite VOVisualizer                                                 #
# ═══════════════════════════════════════════════════════════════════════ #

class VOVisualizer:
    """
    Composite display: camera view (left) + trajectory map (right).
    Call update() each frame; it returns a single BGR composite image.

    Optionally show with cv2.imshow (call `show()`) or save frames.
    """

    def __init__(
        self,
        cam_size  : Tuple[int, int] = (640, 480),
        map_size  : Tuple[int, int] = (500, 480),
        save_path : Optional[str]   = None,
        fps       : int             = 30,
    ):
        self.cam_w, self.cam_h = cam_size
        self.map_w, self.map_h = map_size
        self.traj_plot = TrajectoryPlot(figsize=(map_size[0], map_size[1]))

        self._writer      = None
        if save_path:
            out_w  = self.cam_w + self.map_w
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(save_path, fourcc, fps, (out_w, cam_size[1]))

    def update(
        self,
        frame      : np.ndarray,
        trajectory : np.ndarray,
        map_points : List[MapPoint],
        keyframes  : List[Keyframe],
        hud_info   : Optional[dict] = None,
    ) -> np.ndarray:
        """
        Build and return the composite (camera | trajectory) BGR image.

        hud_info keys: frame_id, process_ms, is_keyframe
        """
        # Camera panel
        cam = _to_bgr(frame)
        cam = cv2.resize(cam, (self.cam_w, self.cam_h))

        if hud_info and len(trajectory) > 0:
            pos = trajectory[-1] if len(trajectory) > 0 else np.zeros(3)
            cam = FeatureOverlay.draw_hud(
                cam,
                frame_id   = hud_info.get("frame_id", 0),
                num_kf     = len(keyframes),
                num_mp     = len(map_points),
                position   = pos,
                is_kf      = hud_info.get("is_keyframe", False),
                process_ms = hud_info.get("process_ms", 0.0),
            )

        # Map panel
        traj_img = self.traj_plot.render_2d(trajectory, map_points, keyframes)
        traj_img = cv2.resize(traj_img, (self.map_w, self.cam_h))

        composite = np.hstack([cam, traj_img])

        if self._writer:
            self._writer.write(composite)

        return composite

    def show(self, composite: np.ndarray, window: str = "Visual Odometry"):
        cv2.imshow(window, composite)

    def release(self):
        if self._writer:
            self._writer.release()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════ #
#  Plotting utilities                                                     #
# ═══════════════════════════════════════════════════════════════════════ #

def plot_trajectory_static(
    trajectory : np.ndarray,
    map_points : List[MapPoint] = [],
    keyframes  : List[Keyframe] = [],
    save_path  : Optional[str]  = None,
    title      : str            = "VO Trajectory",
) -> Figure:
    """
    Create a static publication-quality trajectory figure.
    Returns the Figure (caller can show or save).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))   # hi-res static figure
    fig.patch.set_facecolor("black")

    for ax, (ix, iy, xlabel, ylabel) in [
        (ax1, (0, 2, "X (m)", "Z (m)")),
        (ax2, (0, 1, "X (m)", "Y (m)")),
    ]:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.set_xlabel(xlabel, color="white", weight='bold')
        ax.set_ylabel(ylabel, color="white", weight='bold')
        ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')

        for spine in ax.spines.values():
            spine.set_color("white")

        if map_points:
            mp_xyz = np.array([mp.xyz for mp in map_points])
            ax.scatter(mp_xyz[:, ix], mp_xyz[:, iy],
                       s=0.5, c="yellow", alpha=0.4, linewidths=0)

        if len(trajectory) > 1:
            ax.plot(trajectory[:, ix], trajectory[:, iy],
                    color="red", linewidth=3.5, alpha=1.0, label="Trajectory")
            # Start position
            ax.scatter(trajectory[0, ix], trajectory[0, iy],
                       c='white', marker='s', s=60, zorder=6, label="Start")
            # Current position
            ax.scatter(trajectory[-1, ix], trajectory[-1, iy],
                       c="white", s=80, zorder=5)

        if keyframes:
            pos_x = [kf.T_world_cam[ix, 3] for kf in keyframes]
            pos_y = [kf.T_world_cam[iy, 3] for kf in keyframes]
            ax.scatter(pos_x, pos_y, s=30, c="#4A90D9",
                       zorder=4, label="Keyframes", linewidths=0)

            # Camera-direction arrow on the LATEST keyframe only
            latest = keyframes[-1]
            R = latest.T_world_cam[:3, :3]
            fwd_world = R @ np.array([0.0, 0.0, 1.0])
            dx = fwd_world[ix]
            dy = fwd_world[iy]
            length = max(np.hypot(dx, dy), 1e-6)
            dx /= length
            dy /= length

            xlim = ax.get_xlim() if ax.get_xlim() != (0.0, 1.0) else (-10, 10)
            arrow_len = max(abs(xlim[1] - xlim[0]), 1.0) * 0.04

            ax.annotate(
                "",
                xy=(pos_x[-1] + dx * arrow_len, pos_y[-1] + dy * arrow_len),
                xytext=(pos_x[-1], pos_y[-1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#FFD700",
                    lw=2.0,
                    mutation_scale=14,
                ),
                zorder=7,
            )

        ax.legend(fontsize=9, facecolor="black", labelcolor="white", loc='lower right', edgecolor="white")

    ax1.set_title("Top-down (X–Z)", color="white", fontsize=12, weight='bold')
    ax2.set_title("Front (X–Y)", color="white", fontsize=12, weight='bold')
    fig.suptitle(title, color="white", fontsize=14, weight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ═══════════════════════════════════════════════════════════════════════ #
#  Helpers                                                                #
# ═══════════════════════════════════════════════════════════════════════ #

def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()

def _fig_to_bgr(fig: Figure) -> np.ndarray:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def _axis_indices(axes: str) -> Tuple[int, int]:
    mapping = {"x": 0, "y": 1, "z": 2}
    return mapping[axes[0].lower()], mapping[axes[1].lower()]

def _random_color(seed: int) -> Tuple[int, int, int]:
    rng = np.random.RandomState(seed)
    return tuple(int(c) for c in rng.randint(80, 255, 3))