from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
from .keyframe import Keyframe
from .triangulation import MapPoint


class CovisibilityGraph:
    """
    Undirected weighted graph.
    Nodes  : Keyframe (by kf_id)
    Edges  : weighted by number of shared MapPoints
    
    Used by:
      - Local BA    → get local window of covisible KFs
      - Loop closer → essential graph (edges > 100)
      - Map culling → find redundant KFs
    """

    def __init__(self, min_shared: int = 15):
        self.min_shared = min_shared          # min shared MPs for an edge to exist
        self._kf_map:   Dict[int, Keyframe]  = {}        # kf_id → Keyframe
        self._mp_to_kf: Dict[int, Set[int]]  = defaultdict(set)  # mp_id → {kf_ids}
        self._weights:  Dict[Tuple, int]     = defaultdict(int)   # (kf_a, kf_b) → count

    # ── Add / update ─────────────────────────────────────────────── #

    def add_keyframe(self, kf: Keyframe):
        self._kf_map[kf.kf_id] = kf
        self._update_connections(kf)

    def add_map_point_observation(self, mp_id: int, kf_id: int):
        """Call this when a map point is observed in a keyframe."""
        self._mp_to_kf[mp_id].add(kf_id)
        self._recompute_edges_for(kf_id)

    def _update_connections(self, kf: Keyframe):
        """Recompute all edges touching kf after its map points are set."""
        counter: Dict[int, int] = defaultdict(int)

        for mp in kf.map_points:
            mp_id = id(mp)
            self._mp_to_kf[mp_id].add(kf.kf_id)
            for other_kf_id in self._mp_to_kf[mp_id]:
                if other_kf_id != kf.kf_id:
                    counter[other_kf_id] += 1

        # Update edge weights
        for other_id, count in counter.items():
            key = tuple(sorted((kf.kf_id, other_id)))
            self._weights[key] = count

    def _recompute_edges_for(self, kf_id: int):
        if kf_id in self._kf_map:
            self._update_connections(self._kf_map[kf_id])

    # ── Queries ──────────────────────────────────────────────────── #

    def get_neighbors(self, kf: Keyframe, min_weight: int = None) -> List[Keyframe]:
        """Return covisible keyframes sorted by shared map point count (descending)."""
        threshold = min_weight or self.min_shared
        neighbors = []
        for (a, b), w in self._weights.items():
            if w < threshold:
                continue
            other_id = b if a == kf.kf_id else (a if b == kf.kf_id else None)
            if other_id is not None and other_id in self._kf_map:
                neighbors.append((self._kf_map[other_id], w))
        neighbors.sort(key=lambda x: -x[1])
        return [kf for kf, _ in neighbors]

    def get_local_window(self, kf: Keyframe, max_kfs: int = 20) -> List[Keyframe]:
        """
        Local BA window: current KF + top covisible neighbors.
        Returns list ordered by covisibility weight.
        """
        neighbors = self.get_neighbors(kf)[:max_kfs - 1]
        return [kf] + neighbors

    def get_spanning_tree_edges(self) -> List[Tuple[int, int]]:
        """
        Minimum spanning tree over the graph (by max weight).
        Used by pose-graph optimisation as the backbone edge set.
        """
        if not self._kf_map:
            return []
        # Kruskal's — max weight spanning tree
        sorted_edges = sorted(self._weights.items(), key=lambda x: -x[1])
        parent = {k: k for k in self._kf_map}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        tree = []
        for (a, b), w in sorted_edges:
            if a not in parent or b not in parent:
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
                tree.append((a, b))
        return tree

    def get_essential_graph_edges(self, min_weight: int = 100) -> List[Tuple[int, int]]:
        """Strong covisibility edges — used for pose-graph optimisation."""
        return [(a, b) for (a, b), w in self._weights.items() if w >= min_weight]

    def weight(self, kf_a: Keyframe, kf_b: Keyframe) -> int:
        key = tuple(sorted((kf_a.kf_id, kf_b.kf_id)))
        return self._weights.get(key, 0)

    def __len__(self):
        return len(self._kf_map)