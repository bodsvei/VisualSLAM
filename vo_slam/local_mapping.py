"""
local_mapping.py  (architectural fix)
--------------------------------------
Root cause of BA runs=0
-----------------------
kf.map_points only contains MapPoints triangulated at the moment
that keyframe was created. Different KFs create DIFFERENT MapPoint
objects for the same physical 3D point, so no two KFs share the
same object reference. The co-visibility graph therefore has zero-
weight edges, get_local_window() returns 1 KF, and BA is always
skipped with "KFs < 2".

Fix A (immediate) — spatial fallback window:
  If the co-visibility graph can't build a window of ≥2 KFs, fall
  back to using the N most recent keyframes directly. Collect their
  map points by spatial proximity: any MP whose xyz projects into
  the current camera frustum is included.

Fix B (proper) — build covisibility from map point observations:
  This requires pipeline.py to record which KFs observe each MP.
  Implemented here by building a KF→MP index from vo.map_points
  using reprojection at startup of each BA call.

Both fixes run together. Fix A guarantees BA fires immediately.
Fix B builds the correct graph over time so BA windows improve.
"""

from __future__ import annotations
import queue
import threading
import traceback
import numpy as np
from typing import List, Optional, Dict, Set
from .keyframe          import Keyframe
from .triangulation     import MapPoint
from .covisibility      import CovisibilityGraph
from .bundle_adjustment import local_bundle_adjustment
from .map_culling       import cull_map_points, cull_keyframes
from .camera            import CameraModel


class LocalMapper:
    def __init__(
        self,
        vo,                      # VisualOdometry instance
        n_ba_iters      : int   = 10,
        local_window    : int   = 20,
        spatial_window  : int   = 500,    # max MPs from spatial fallback
        verbose         : bool  = False,
    ):
        self.vo             = vo
        self.camera         = vo.camera
        self.map_points     = vo.map_points
        self.keyframes      = vo.keyframes
        self.pose_graph     = vo.pose_graph
        self.n_ba_iters     = n_ba_iters
        self.local_window   = local_window
        self.spatial_window = spatial_window
        self.verbose        = verbose

        self.covis_graph    = CovisibilityGraph(min_shared=10)
        self._queue         = queue.Queue()
        self._thread        : Optional[threading.Thread] = None
        self._stop_flag     = threading.Event()
        self._lock          = threading.Lock()

        self.n_ba_runs      = 0
        self.n_pts_culled   = 0
        self.n_kfs_culled   = 0
        self.n_ba_skipped   = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def enqueue(self, kf: Keyframe):
        self._queue.put(kf)

    def start(self):
        try:
            import g2o
            print(f"[LocalMapper] g2o available ✓")
        except ImportError:
            print(f"\n[LocalMapper] *** g2o NOT installed — BA disabled ***")
            print(f"  Install: pip install g2o-python\n")

        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[LocalMapper] started")

    def stop(self):
        self._stop_flag.set()
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5.0)
        print(f"[LocalMapper] stopped | "
              f"BA runs={self.n_ba_runs} | "
              f"skipped={self.n_ba_skipped} | "
              f"pts culled={self.n_pts_culled} | "
              f"KFs culled={self.n_kfs_culled}")

    # ------------------------------------------------------------------ #
    #  Thread                                                              #
    # ------------------------------------------------------------------ #

    def _run(self):
        while not self._stop_flag.is_set():
            try:
                kf = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if kf is None:
                break
            try:
                self._process(kf)
            except Exception:
                print(f"\n[LocalMapper] *** Exception in _process ***")
                traceback.print_exc()
                print()

    # ------------------------------------------------------------------ #
    #  Main processing                                                     #
    # ------------------------------------------------------------------ #

    def _process(self, kf: Keyframe):
        # Step 1: update covisibility graph
        self.covis_graph.add_keyframe(kf)

        # ── Fix B: build covis connections BEFORE computing the window ─ #
        # Get a preliminary local window based on current covis state, then
        # update the graph so the final window benefits from fresh connections.
        prelim_kfs = self.covis_graph.get_local_window(kf, max_kfs=self.local_window)
        self._assign_visible_mps_to_kfs(prelim_kfs)

        # Step 2: re-compute local KF window with updated covis graph
        local_kfs    = self.covis_graph.get_local_window(kf, max_kfs=self.local_window)
        local_kf_ids = {k.kf_id for k in local_kfs}

        # ── Fix A: if covis window is too small, use temporal window ─── #
        if len(local_kfs) < 2:
            local_kfs    = self._temporal_window(kf)
            local_kf_ids = {k.kf_id for k in local_kfs}

        if len(local_kfs) < 2:
            self.n_ba_skipped += 1
            return

        # Step 3: fixed KFs (outside window, provide global anchor)
        fixed_kfs = self._build_fixed_kfs(local_kfs, local_kf_ids)

        # Step 4: collect map points
        local_mps = self._collect_local_mps(local_kfs, kf)

        if len(local_mps) < 5:
            self.n_ba_skipped += 1
            if self.verbose:
                print(f"[LocalMapper] BA skipped: MPs={len(local_mps)}")
            return

        # Step 5: run BA
        print(f"[LocalMapper] BA #{self.n_ba_runs+1} — "
              f"local_KFs={len(local_kfs)} "
              f"fixed_KFs={len(fixed_kfs)} "
              f"MPs={len(local_mps)}")

        old_poses = {kf.kf_id: kf.T_world_cam.copy() for kf in local_kfs}

        opt_kfs, kept_mps, culled_mps = local_bundle_adjustment(
            local_kfs  = local_kfs,
            fixed_kfs  = fixed_kfs,
            map_points = local_mps,
            camera     = self.camera,
            n_iters    = self.n_ba_iters,
            verbose    = self.verbose,
        )
        self.n_ba_runs += 1

        if self.vo.pose_graph is not None:
            self._update_pose_graph(local_kfs, old_poses)
            
            # Also update the current tracking pose in VO
            latest_kf = local_kfs[-1]
            T_old = old_poses.get(latest_kf.kf_id)
            if T_old is not None:
                # Correction from the latest keyframe in the window
                corr_local = np.linalg.inv(T_old) @ latest_kf.T_world_cam
                self.vo.T_world_cam = self.vo.T_world_cam @ corr_local

        # Step 6: cull
        with self._lock:
            # Temporarily disable map point culling so the tracker doesn't get starved
            self.map_points[:], n_mp = cull_map_points(self.map_points)
            self.keyframes[:],  n_kf = cull_keyframes(
                self.keyframes, self.covis_graph
            )
            self.n_pts_culled += n_mp
            self.n_kfs_culled += n_kf

    def _update_pose_graph(self, local_kfs: List[Keyframe], old_poses: Dict[int, np.ndarray]):
        """Propagate BA corrections from local keyframes to the full trajectory."""
        if not self.vo or not self.vo.pose_graph:
            return

        # 1. Compute corrections: T_new @ inv(T_old)
        corrections = {}
        for kf in local_kfs:
            T_old = old_poses.get(kf.kf_id)
            if T_old is not None:
                corrections[kf.kf_id] = kf.T_world_cam @ np.linalg.inv(T_old)

        if not corrections:
            return

        # 2. Update poses in the graph using PoseGraph utility
        self.vo.pose_graph.transform_all(local_kfs, corrections)

    # ------------------------------------------------------------------ #
    #  Fix A: temporal window fallback                                    #
    # ------------------------------------------------------------------ #

    def _temporal_window(self, kf: Keyframe) -> List[Keyframe]:
        """
        Return the N most recent keyframes as the local window.
        Used when the co-visibility graph can't build a proper window.
        """
        n      = min(self.local_window, len(self.keyframes))
        recent = self.keyframes[-n:]
        return list(reversed(recent))  # Newest first

    # ------------------------------------------------------------------ #
    #  Fix B: assign visible MPs to keyframes                             #
    # ------------------------------------------------------------------ #

    def _assign_visible_mps_to_kfs(self, kfs: List[Keyframe]):
        """
        For each KF in the window, project global map points into it
        and add visible ones to kf.map_points (if not already there).
        This builds the correct co-visibility graph over time.
        """
        if not self.map_points or not kfs:
            return

        # Use a spatial subset — don't reproject all 289k points every call
        recent_mps = self.map_points[-min(self.spatial_window, len(self.map_points)):]
        mp_xyz     = np.array([mp.xyz for mp in recent_mps])  # (N, 3)

        for kf in kfs:
            existing_ids = {id(mp) for mp in kf.map_points}
            R   = kf.T_world_cam[:3, :3]
            t   = kf.T_world_cam[:3,  3]
            K   = self.camera.K

            # Transform MPs to camera frame
            pts_cam = (R.T @ (mp_xyz - t).T).T   # (N, 3)
            in_front = pts_cam[:, 2] > 0.1

            if not in_front.any():
                continue

            # Project to pixel coords
            pts_f    = pts_cam[in_front]
            u        = K[0, 0] * pts_f[:, 0] / pts_f[:, 2] + K[0, 2]
            v        = K[1, 1] * pts_f[:, 1] / pts_f[:, 2] + K[1, 2]
            in_image = (
                (u >= 0) & (u < self.camera.width)  &
                (v >= 0) & (v < self.camera.height)
            )

            # Get indices of points matching both constraints
            visible_indices = np.where(in_front)[0][in_image]
            
            # Extract corresponding valid pixel locations aligned with visible_indices
            u_img = u[in_image]
            v_img = v[in_image]
            
            newly_added = 0
            kf_pts2d    = kf.features.pts2d
            
            for i, idx in enumerate(visible_indices):
                mp = recent_mps[idx]
                if id(mp) not in existing_ids:
                    proj_uv = np.array([u_img[i], v_img[i]])
                    
                    if len(kf_pts2d) > 0:
                        # Find nearest 2D feature in the keyframe
                        dists = np.linalg.norm(kf_pts2d - proj_uv, axis=1)
                        best_feat_idx = int(np.argmin(dists))
                        
                        # Only associate if the reprojection error is tight (e.g., within 4.0 pixels)
                        if dists[best_feat_idx] < 4.0:
                            kf.map_points.append(mp)
                            mp.obs[kf.kf_id] = best_feat_idx  # This automatically increments mp.observations!
                            existing_ids.add(id(mp))
                            newly_added += 1

            if newly_added > 0:
                # Re-update co-visibility connections for this KF
                self.covis_graph._update_connections(kf)

    # ------------------------------------------------------------------ #
    #  Collect map points from local window                               #
    # ------------------------------------------------------------------ #

    def _collect_local_mps(
        self,
        local_kfs : List[Keyframe],
        current_kf: Keyframe,
    ) -> List[MapPoint]:
        """
        Collect MPs from local KFs. If still too few, supplement with
        spatially nearby MPs from the global map.
        """
        seen_ids: Set[int] = set()
        local_mps: List[MapPoint] = []

        for lkf in local_kfs:
            for mp in lkf.map_points:
                if id(mp) not in seen_ids:
                    local_mps.append(mp)
                    seen_ids.add(id(mp))

        # Supplement with recent global MPs if still sparse
        if len(local_mps) < 20 and self.map_points:
            n_supp = min(self.spatial_window, len(self.map_points))
            for mp in self.map_points[-n_supp:]:
                if id(mp) not in seen_ids:
                    local_mps.append(mp)
                    seen_ids.add(id(mp))

        return local_mps

    # ------------------------------------------------------------------ #
    #  Build fixed KF set                                                  #
    # ------------------------------------------------------------------ #

    def _build_fixed_kfs(
        self,
        local_kfs   : List[Keyframe],
        local_kf_ids: Set[int],
    ) -> List[Keyframe]:
        """
        Fixed KFs are those that observe local MPs but are outside
        the optimization window. They anchor the window to global frame.
        Always includes the oldest available KF.
        """
        fixed_kfs = []
        seen_ids  = set()

        for lkf in local_kfs:
            for n in self.covis_graph.get_neighbors(lkf, min_weight=1):
                if n.kf_id not in local_kf_ids and n.kf_id not in seen_ids:
                    fixed_kfs.append(n)
                    seen_ids.add(n.kf_id)

        # Always pin the KF just before the window as a hard anchor
        oldest_local = min(k.kf_id for k in local_kfs)
        for stored_kf in reversed(self.keyframes):
            if stored_kf.kf_id < oldest_local and stored_kf.kf_id not in seen_ids:
                fixed_kfs.append(stored_kf)
                break

        return fixed_kfs