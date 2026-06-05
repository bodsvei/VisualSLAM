"""
LocalMapper — runs in a background thread.
Receives keyframes from VO via a queue and runs:
  1. Co-visibility update
  2. Local Bundle Adjustment
  3. Map culling
"""

import queue
import threading
import numpy as np
from typing import Optional, List
from .keyframe      import Keyframe
from .triangulation import MapPoint
from .covisibility  import CovisibilityGraph
from .bundle_adjustment import local_bundle_adjustment
from .map_culling   import cull_map_points, cull_keyframes
from .camera        import CameraModel


class LocalMapper:
    """
    Usage
    -----
    mapper = LocalMapper(camera)
    vo.on_new_keyframe = mapper.enqueue   # hook into VO

    mapper.start()    # spawns background thread
    ...
    mapper.stop()
    """

    def __init__(
        self,
        camera          : CameraModel,
        map_points_ref  : List[MapPoint],   # shared reference to vo.map_points
        keyframes_ref   : List[Keyframe],   # shared reference to vo.keyframes
        n_ba_iters      : int   = 10,
        local_window    : int   = 20,
        verbose         : bool  = False,
    ):
        self.camera         = camera
        self.map_points     = map_points_ref
        self.keyframes      = keyframes_ref
        self.n_ba_iters     = n_ba_iters
        self.local_window   = local_window
        self.verbose        = verbose

        self.covis_graph    = CovisibilityGraph()
        self._queue         = queue.Queue()
        self._thread        : Optional[threading.Thread] = None
        self._stop_flag     = threading.Event()
        self._lock          = threading.Lock()   # guards map_points + keyframes

        self.n_ba_runs      = 0
        self.n_pts_culled   = 0
        self.n_kfs_culled   = 0

    def enqueue(self, kf: Keyframe):
        """Called by VO on_new_keyframe hook — non-blocking."""
        self._queue.put(kf)

    def start(self):
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[LocalMapper] started")

    def stop(self):
        self._stop_flag.set()
        self._queue.put(None)   # unblock queue.get()
        if self._thread:
            self._thread.join(timeout=5.0)
        print(f"[LocalMapper] stopped | "
              f"BA runs={self.n_ba_runs} | "
              f"pts culled={self.n_pts_culled} | "
              f"KFs culled={self.n_kfs_culled}")

    def _run(self):
        while not self._stop_flag.is_set():
            try:
                kf = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if kf is None:
                break
            self._process(kf)

    def _process(self, kf: Keyframe):
        # 1. Update co-visibility graph
        self.covis_graph.add_keyframe(kf)

        # 2. Get local window
        local_kfs = self.covis_graph.get_local_window(kf, max_kfs=self.local_window)
        local_kf_ids = {k.kf_id for k in local_kfs}

        # Fixed KFs: observe local MPs but outside the window
        fixed_kfs = []
        seen_ids  = set()
        for lkf in local_kfs:
            neighbors = self.covis_graph.get_neighbors(lkf, min_weight=1)
            for n in neighbors:
                if n.kf_id not in local_kf_ids and n.kf_id not in seen_ids:
                    fixed_kfs.append(n)
                    seen_ids.add(n.kf_id)

        # Collect map points visible in local window
        local_mps = []
        seen_mp_ids = set()
        for lkf in local_kfs:
            for mp in lkf.map_points:
                if id(mp) not in seen_mp_ids:
                    local_mps.append(mp)
                    seen_mp_ids.add(id(mp))

        if len(local_mps) < 10 or len(local_kfs) < 2:
            return

        # 3. Local Bundle Adjustment
        opt_kfs, kept_mps, culled_mps = local_bundle_adjustment(
            local_kfs  = local_kfs,
            fixed_kfs  = fixed_kfs,
            map_points = local_mps,
            camera     = self.camera,
            n_iters    = self.n_ba_iters,
            verbose    = self.verbose,
        )
        self.n_ba_runs += 1

        # 4. Map culling
        with self._lock:
            self.map_points[:], n_mp_culled = cull_map_points(self.map_points)
            self.keyframes[:],  n_kf_culled = cull_keyframes(
                self.keyframes, self.covis_graph
            )
            self.n_pts_culled += n_mp_culled
            self.n_kfs_culled += n_kf_culled

        if self.verbose:
            print(f"  [BA] KFs={len(local_kfs)} MPs={len(local_mps)} "
                  f"culled_pts={n_mp_culled} culled_kfs={n_kf_culled}")