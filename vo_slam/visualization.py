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
        figsize    : Tuple[int, int] = (600, 600),
        point_size : float = 0.5,
        traj_color : str   = "#00e676",
        kf_color   : str   = "#ff5722",
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
        fig, ax = plt.subplots(figsize=(self.fig_w/100, self.fig_h/100), dpi=100)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        ix, iy = _axis_indices(axes)

        # Map points
        if map_points:
            mp_xyz = np.array([mp.xyz for mp in map_points])
            ax.scatter(mp_xyz[:, ix], mp_xyz[:, iy],
                       s=self.point_size, c="#546e7a", alpha=0.4, linewidths=0)

        # Trajectory
        if len(trajectory) > 1:
            ax.plot(trajectory[:, ix], trajectory[:, iy],
                    color=self.traj_color, linewidth=1.2, alpha=0.9, label="Trajectory")
            ax.scatter(trajectory[-1, ix], trajectory[-1, iy],
                       c=self.traj_color, s=40, zorder=5)

        # Keyframe positions
        if keyframes:
            kf_pos = np.array([kf.position for kf in keyframes])
            ax.scatter(kf_pos[:, ix], kf_pos[:, iy],
                       c=self.kf_color, s=20, zorder=4, label="Keyframes")

        ax.set_xlabel(axes[0].upper(), color="white")
        ax.set_ylabel(axes[1].upper(), color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.set_title("VO Trajectory", color="white", fontsize=10)
        ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")
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
        fig.patch.set_facecolor("#1a1a2e")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#16213e")

        if map_points:
            mp_xyz = np.array([mp.xyz for mp in map_points])
            ax.scatter(mp_xyz[:, 0], mp_xyz[:, 1], mp_xyz[:, 2],
                       s=self.point_size, c="#546e7a", alpha=0.3)

        if len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                    color=self.traj_color, linewidth=1.5)
            ax.scatter(*trajectory[-1], c=self.traj_color, s=60)

        if keyframes:
            kf_pos = np.array([kf.position for kf in keyframes])
            ax.scatter(kf_pos[:, 0], kf_pos[:, 1], kf_pos[:, 2],
                       c=self.kf_color, s=25)

        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.set_zlabel("Z", color="white")
        ax.set_title("VO Trajectory 3D", color="white")

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")

    for ax, (ix, iy, xlabel, ylabel) in [
        (ax1, (0, 2, "X (m)", "Z (m)")),
        (ax2, (0, 1, "X (m)", "Y (m)")),
    ]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.set_xlabel(xlabel, color="white")
        ax.set_ylabel(ylabel, color="white")
        for spine in ax.spines.values():
            spine.set_color("#444")

        if map_points:
            mp_xyz = np.array([mp.xyz for mp in map_points])
            ax.scatter(mp_xyz[:, ix], mp_xyz[:, iy],
                       s=0.3, c="#546e7a", alpha=0.3)

        if len(trajectory) > 1:
            ax.plot(trajectory[:, ix], trajectory[:, iy],
                    color="#00e676", linewidth=1.5, label="Trajectory")

        if keyframes:
            kf_pos = np.array([kf.position for kf in keyframes])
            ax.scatter(kf_pos[:, ix], kf_pos[:, iy],
                       c="#ff5722", s=15, zorder=5, label="Keyframes")

        ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

    ax1.set_title("Top-down  (X–Z)", color="white")
    ax2.set_title("Front     (X–Y)", color="white")
    fig.suptitle(title, color="white", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight",
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
