"""
map_storage.py
--------------
Save and load the full SLAM map to/from disk.

Serialises
----------
  - All keyframes (poses, feature pts2d, descriptors)
  - All map points (xyz, observations)
  - BoW database (vocabulary + inverted index)
  - Camera intrinsics
  - PoseGraph poses

Format: single .pkl file (fast, no external dependencies)
        or a .npz + .pkl split for large maps

Usage
-----
  # Save after a run:
  MapStorage.save("map_lab.pkl", vo, loop_detector)

  # Load for a new session:
  loaded = MapStorage.load("map_lab.pkl")
  # loaded.keyframes, loaded.map_points, loaded.recognizer, loaded.camera
"""

from __future__ import annotations
import pickle
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from datetime import datetime


@dataclass
class SavedMap:
    """Container for a fully loaded map."""
    camera_K       : np.ndarray            # 3×3
    camera_wh      : tuple                 # (width, height)
    keyframes      : list                  # List[SavedKeyframe]
    map_points     : list                  # List[SavedMapPoint]
    poses          : List[np.ndarray]      # full pose graph (T_world_cam per frame)
    recognizer     : object                # PlaceRecognizer (with vocab + DB)
    metadata       : dict = field(default_factory=dict)

    @property
    def n_keyframes(self):
        return len(self.keyframes)

    @property
    def n_map_points(self):
        return len(self.map_points)


@dataclass
class SavedKeyframe:
    kf_id       : int
    frame_id    : int
    T_world_cam : np.ndarray    # 4×4
    pts2d       : np.ndarray    # (N, 2) float32
    descriptors : np.ndarray    # (N, 32) uint8
    timestamp   : float = 0.0


@dataclass
class SavedMapPoint:
    xyz        : np.ndarray    # (3,)
    ref_kf_id  : int
    reproj_err : float
    observations: int


class MapStorage:
    """Save and load the full SLAM map."""

    # ------------------------------------------------------------------ #
    #  Save                                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def save(
        path          : str,
        vo,                      # VisualOdometry instance
        loop_detector = None,    # LoopDetector instance (optional, for vocab+DB)
        verbose       : bool = True,
    ) -> None:
        """
        Serialise the full map to a .pkl file.

        Parameters
        ----------
        path          : output file path  (e.g. "map_session1.pkl")
        vo            : VisualOdometry instance after a run
        loop_detector : LoopDetector instance (carries vocab + BoW DB)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialise keyframes
        saved_kfs = []
        for kf in vo.keyframes:
            saved_kfs.append(SavedKeyframe(
                kf_id       = kf.kf_id,
                frame_id    = kf.frame_id,
                T_world_cam = kf.T_world_cam.copy(),
                pts2d       = kf.features.pts2d.copy(),
                descriptors = kf.features.descriptors.copy()
                              if kf.features.descriptors is not None
                              else np.empty((0, 32), dtype=np.uint8),
                timestamp   = kf.timestamp,
            ))

        # Serialise map points
        saved_mps = []
        for mp in vo.map_points:
            saved_mps.append(SavedMapPoint(
                xyz         = mp.xyz.copy(),
                ref_kf_id   = mp.ref_idx,
                reproj_err  = mp.reproj_err,
                observations= mp.observations,
            ))

        # Serialise recognizer (vocab + DB) if available
        recognizer = None
        if loop_detector is not None and loop_detector._recognizer is not None:
            recognizer = loop_detector._recognizer

        data = SavedMap(
            camera_K    = vo.camera.K.copy(),
            camera_wh   = (vo.camera.width, vo.camera.height),
            keyframes   = saved_kfs,
            map_points  = saved_mps,
            poses       = [T.copy() for T in vo.pose_graph.poses],
            recognizer  = recognizer,
            metadata    = {
                "saved_at"     : datetime.now().isoformat(),
                "n_frames"     : vo.frame_id,
                "n_keyframes"  : len(saved_kfs),
                "n_map_points" : len(saved_mps),
                "final_state"  : vo.state.name,
            },
        )

        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=4)

        size_mb = path.stat().st_size / 1e6
        if verbose:
            print(f"[MapStorage] Saved → {path}  "
                  f"({len(saved_kfs)} KFs, {len(saved_mps)} MPs, {size_mb:.1f} MB)")

    # ------------------------------------------------------------------ #
    #  Load                                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def load(path: str, verbose: bool = True) -> SavedMap:
        """
        Load a saved map from disk.

        Returns a SavedMap — pass it to Relocalization or
        continue mapping from it.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Map file not found: {path}")

        with open(path, "rb") as f:
            data: SavedMap = pickle.load(f)

        if verbose:
            meta = data.metadata
            print(f"[MapStorage] Loaded {path}")
            print(f"             Saved at    : {meta.get('saved_at', 'unknown')}")
            print(f"             Keyframes   : {data.n_keyframes}")
            print(f"             Map points  : {data.n_map_points}")
            print(f"             Frames total: {meta.get('n_frames', '?')}")

        return data

    # ------------------------------------------------------------------ #
    #  Info                                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def info(path: str) -> None:
        """Print a summary of a saved map without fully loading it."""
        data = MapStorage.load(path, verbose=False)
        print(f"Map file      : {path}")
        print(f"Saved at      : {data.metadata.get('saved_at', '?')}")
        print(f"Keyframes     : {data.n_keyframes}")
        print(f"Map points    : {data.n_map_points}")
        print(f"Pose entries  : {len(data.poses)}")
        print(f"Has vocab/DB  : {data.recognizer is not None}")
        print(f"Camera K      :\n{data.camera_K}")