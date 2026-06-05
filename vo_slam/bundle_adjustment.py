"""
Local Bundle Adjustment using g2o.

Optimises:
  - Poses of KFs in the local window
  - All map points observed by those KFs

KFs outside the local window that observe local map points
are added as FIXED vertices (provide constraints, not optimised).
"""

import numpy as np
from typing import List, Set
from .keyframe import Keyframe
from .triangulation import MapPoint
from .camera import CameraModel

try:
    import g2o
    G2O_AVAILABLE = True
except ImportError:
    G2O_AVAILABLE = False
    print("[BA] g2o not available — bundle adjustment disabled")


def local_bundle_adjustment(
    local_kfs  : List[Keyframe],    # KFs to optimise (from covis window)
    fixed_kfs  : List[Keyframe],    # KFs to fix (outside window but share MPs)
    map_points : List[MapPoint],    # all map points visible in local_kfs
    camera     : CameraModel,
    n_iters    : int = 10,
    verbose    : bool = False,
) -> tuple[List[Keyframe], List[MapPoint], List[MapPoint]]:
    """
    Returns (optimised_kfs, kept_map_points, culled_map_points).
    If g2o unavailable, returns inputs unchanged.
    """
    if not G2O_AVAILABLE:
        return local_kfs, map_points, []

    optimizer = _build_optimizer()

    kf_vertex_ids: dict  = {}    # kf_id → g2o vertex id
    mp_vertex_ids: dict  = {}    # mp object id → g2o vertex id
    next_id = 0

    # ── Add KF pose vertices ─────────────────────────────────────── #
    for kf in local_kfs + fixed_kfs:
        v = g2o.VertexSE3Expmap()
        v.set_id(next_id)
        R = kf.T_world_cam[:3, :3]
        t = kf.T_world_cam[:3,  3]
        # g2o uses T_cam_world (inverse of our convention)
        R_cw = R.T
        t_cw = -R.T @ t
        v.set_estimate(g2o.SE3Quat(R_cw, t_cw))
        v.set_fixed(kf in fixed_kfs or kf.kf_id == 0)  # always fix origin
        optimizer.add_vertex(v)
        kf_vertex_ids[kf.kf_id] = next_id
        next_id += 1

    # ── Add map point vertices ───────────────────────────────────── #
    for mp in map_points:
        mp_id = id(mp)
        if mp_id in mp_vertex_ids:
            continue
        v = g2o.VertexPointXYZ()
        v.set_id(next_id)
        v.set_estimate(mp.xyz)
        v.set_marginalized(True)
        optimizer.add_vertex(v)
        mp_vertex_ids[mp_id] = next_id
        next_id += 1

    # ── Add reprojection edges ───────────────────────────────────── #
    edges = []
    local_kf_ids: Set[int] = {kf.kf_id for kf in local_kfs}

    for kf in local_kfs + fixed_kfs:
        if kf.kf_id not in kf_vertex_ids:
            continue
        for mp in kf.map_points:
            mp_id = id(mp)
            if mp_id not in mp_vertex_ids:
                continue

            # Get observation (pixel coordinate)
            obs_idx = mp.ref_idx if kf.kf_id == 0 else mp.cur_idx
            if obs_idx >= len(kf.features.pts2d):
                continue
            uv = kf.features.pts2d[obs_idx]

            e = g2o.EdgeSE3ProjectXYZ()
            e.set_vertex(0, optimizer.vertex(mp_vertex_ids[mp_id]))
            e.set_vertex(1, optimizer.vertex(kf_vertex_ids[kf.kf_id]))
            e.set_measurement(uv)
            e.set_information(np.eye(2))

            rk = g2o.RobustKernelHuber()
            rk.set_delta(np.sqrt(5.991))   # 95% chi2 confidence, 2 DOF
            e.set_robust_kernel(rk)

            e.fx = camera.fx
            e.fy = camera.fy
            e.cx = camera.cx
            e.cy = camera.cy

            optimizer.add_edge(e)
            edges.append((e, mp, kf.kf_id in local_kf_ids))

    if len(edges) < 10:
        return local_kfs, map_points, []

    # ── Optimise ─────────────────────────────────────────────────── #
    optimizer.initialize_optimization()
    optimizer.set_verbose(verbose)
    optimizer.optimize(n_iters)

    # ── Recover optimised KF poses ───────────────────────────────── #
    for kf in local_kfs:
        v   = optimizer.vertex(kf_vertex_ids[kf.kf_id])
        se3 = v.estimate()
        R_cw = se3.rotation().matrix()
        t_cw = se3.translation()
        # Convert back to T_world_cam
        T = np.eye(4)
        T[:3, :3] = R_cw.T
        T[:3,  3] = -R_cw.T @ t_cw
        kf.T_world_cam = T

    # ── Recover optimised map point positions ────────────────────── #
    kept, culled = [], []
    reproj_threshold = 5.991   # chi2 95%, 2 DOF

    for e, mp, is_local in edges:
        mp_id = id(mp)
        if mp_id not in mp_vertex_ids:
            continue
        v = optimizer.vertex(mp_vertex_ids[mp_id])
        mp.xyz = v.estimate().copy()

        # Cull by final reprojection error
        if e.chi2() > reproj_threshold and is_local:
            culled.append(mp)
        else:
            kept.append(mp)

    culled_set = set(id(m) for m in culled)
    kept = [mp for mp in map_points if id(mp) not in culled_set]
    return local_kfs, kept, culled


def _build_optimizer():
    optimizer = g2o.SparseOptimizer()
    solver    = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)
    return optimizer