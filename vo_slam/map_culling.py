"""
Map Culling — keeps the map lean and real-time capable.

Two culling policies:
  1. Map point culling  — remove weak / poorly observed points
  2. Keyframe culling   — remove redundant keyframes
"""

from typing import List, Tuple
from .keyframe import Keyframe
from .triangulation import MapPoint
from .covisibility import CovisibilityGraph


def cull_map_points(
    map_points      : List[MapPoint],
    min_observations: int   = 3,
    max_reproj_err  : float = 3.0,
) -> Tuple[List[MapPoint], int]:
    """
    Remove map points that are:
      - Seen in fewer than min_observations keyframes
      - Have high reprojection error

    Returns (kept, n_culled).
    """
    kept, n_culled = [], 0
    for mp in map_points:
        if mp.observations < min_observations:
            n_culled += 1
            continue
        if mp.reproj_err > max_reproj_err:
            n_culled += 1
            continue
        kept.append(mp)
    return kept, n_culled


def cull_keyframes(
    keyframes   : List[Keyframe],
    covis_graph : CovisibilityGraph,
    redundancy  : float = 0.9,    # KF is redundant if 90%+ of its MPs are
    min_kfs     : int   = 5,      # never cull below this many KFs
) -> Tuple[List[Keyframe], int]:
    """
    Remove keyframes where >= `redundancy` fraction of their map points
    are observed by at least 3 other keyframes in the covisibility graph.

    Returns (kept_keyframes, n_culled).
    Never culls the first KF or the most recent KF.
    """
    if len(keyframes) <= min_kfs:
        return keyframes, 0

    kept, n_culled = [], 0
    recent_kf_id   = keyframes[-1].kf_id

    for kf in keyframes:
        # Never cull origin or most recent keyframe
        if kf.kf_id == 0 or kf.kf_id == recent_kf_id:
            kept.append(kf)
            continue

        if not kf.map_points:
            kept.append(kf)
            continue

        # Count how many of this KF's MPs are covered by >= 3 other KFs
        neighbors = covis_graph.get_neighbors(kf, min_weight=1)
        neighbor_ids = {n.kf_id for n in neighbors}

        covered = 0
        for mp in kf.map_points:
            # How many OTHER keyframes also see this map point?
            observers = sum(
                1 for n in neighbors
                if any(id(m) == id(mp) for m in n.map_points)
            )
            if observers >= 3:
                covered += 1

        ratio = covered / len(kf.map_points)
        if ratio >= redundancy:
            n_culled += 1
            # Don't add to kept — effectively removes from active map
        else:
            kept.append(kf)

    return kept, n_culled