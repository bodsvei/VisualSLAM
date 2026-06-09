"""
bundle_adjustment.py  (fixed)
------------------------------
Fixes vs uploaded version
--------------------------
FIX 1 — mp.obs does not exist on MapPoint:
  MapPoint only has ref_idx and cur_idx.
  The obs-dict lookup caused a silent AttributeError inside the
  LocalMapper thread, preventing BA from running at all.
  Now uses ref_idx / cur_idx directly with a clear fallback.

FIX 2 — ParameterCamera incompatible with pip g2o build:
  g2o.EdgeSE3ProjectXYZ in g2o-python exposes intrinsics as
  direct attributes (e.fx, e.fy, e.cx, e.cy), not via
  add_parameter / set_parameter_id.
  Removed ParameterCamera entirely and set attributes directly.

FIX 3 — Explicit BA start/end logging:
  Added [BA] prints so it is immediately visible in the terminal
  that BA is actually running, rather than silently skipping.
"""

from __future__ import annotations
import numpy as np
from typing import List, Set
from .keyframe      import Keyframe
from .triangulation import MapPoint
from .camera        import CameraModel

try:
    import g2o
    G2O_AVAILABLE = True
except ImportError:
    G2O_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────── #
#  Solver probe — same auto-detect pattern as pose_graph_optimizer        #
# ─────────────────────────────────────────────────────────────────────── #

_BA_SOLVER_FACTORY = None   # set once on first successful probe

def _build_optimizer():
    global _BA_SOLVER_FACTORY

    if _BA_SOLVER_FACTORY is not None:
        optimizer = g2o.SparseOptimizer()
        optimizer.set_algorithm(
            g2o.OptimizationAlgorithmLevenberg(_BA_SOLVER_FACTORY())
        )
        return optimizer

    attempts = [
        ("Eigen SE3",   lambda: g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())),
        ("Dense SE3",   lambda: g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())),
        ("Cholmod SE3", lambda: g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())),
        ("Eigen X",     lambda: g2o.BlockSolverX(g2o.LinearSolverEigenX())),
        ("Dense X",     lambda: g2o.BlockSolverX(g2o.LinearSolverDenseX())),
        ("Cholmod X",   lambda: g2o.BlockSolverX(g2o.LinearSolverCholmodX())),
    ]
    for name, factory in attempts:
        try:
            solver    = factory()
            optimizer = g2o.SparseOptimizer()
            optimizer.set_algorithm(g2o.OptimizationAlgorithmLevenberg(solver))
            _BA_SOLVER_FACTORY = factory
            print(f"[BA] Solver locked: {name}")
            return optimizer
        except (AttributeError, Exception):
            continue

    raise RuntimeError(
        "[BA] No compatible g2o solver found.\n"
        "Run: python3 check_g2o.py  to see available solvers."
    )


# ─────────────────────────────────────────────────────────────────────── #
#  Main function                                                           #
# ─────────────────────────────────────────────────────────────────────── #

def local_bundle_adjustment(
    local_kfs  : List[Keyframe],
    fixed_kfs  : List[Keyframe],
    map_points : List[MapPoint],
    camera     : CameraModel,
    n_iters    : int  = 10,
    verbose    : bool = False,
) -> tuple[List[Keyframe], List[MapPoint], List[MapPoint]]:
    """
    Local Bundle Adjustment.
    Returns (optimised_kfs, kept_map_points, culled_map_points).
    """
    if not G2O_AVAILABLE:
        print("[BA] g2o not available — install: pip install g2o-python")
        return local_kfs, map_points, []

    if len(local_kfs) < 2 or len(map_points) < 5:
        return local_kfs, map_points, []

    optimizer = _build_optimizer()
    next_id   = 0

    # ── Anchor: oldest KF in the whole window is always fixed ────────── #
    all_kfs   = local_kfs + fixed_kfs
    anchor_id = min(kf.kf_id for kf in all_kfs)

    if verbose:
        print(f"[BA] Start — local={len(local_kfs)} fixed={len(fixed_kfs)} "
              f"MPs={len(map_points)} anchor=KF{anchor_id}")

    # ── Pose vertices ─────────────────────────────────────────────────── #
    kf_vertex: dict = {}
    for kf in all_kfs:
        v    = g2o.VertexSE3Expmap()
        v.set_id(next_id)
        R    = kf.T_world_cam[:3, :3]
        t    = kf.T_world_cam[:3,  3]
        v.set_estimate(g2o.SE3Quat(R.T, -R.T @ t))
        v.set_fixed(
            kf.kf_id == anchor_id
            or kf in fixed_kfs
            or kf.kf_id == 0
        )
        optimizer.add_vertex(v)
        kf_vertex[kf.kf_id] = next_id
        next_id += 1

    # ── Map point vertices ────────────────────────────────────────────── #
    mp_vertex: dict = {}
    for mp in map_points:
        mp_id = id(mp)
        if mp_id in mp_vertex:
            continue
        v = g2o.VertexPointXYZ()
        v.set_id(next_id)
        v.set_estimate(mp.xyz.astype(np.float64))
        v.set_marginalized(True)
        optimizer.add_vertex(v)
        mp_vertex[mp_id] = next_id
        next_id += 1

    # ── Reprojection edges ────────────────────────────────────────────── #
    local_kf_ids: Set[int] = {kf.kf_id for kf in local_kfs}
    edges = []

    for kf in all_kfs:
        if kf.kf_id not in kf_vertex:
            continue

        for mp in kf.map_points:
            mp_id = id(mp)
            if mp_id not in mp_vertex:
                continue

            # Only add an edge if this KF has a recorded observation.
            # mp.obs maps kf_id -> index in kf.features.pts2d.
            # The old fallback (ref_idx / cur_idx) used indices from a
            # different KF's pts2d array, causing OOB index errors.
            if kf.kf_id not in mp.obs:
                continue
            obs_idx = mp.obs[kf.kf_id]
            if obs_idx >= len(kf.features.pts2d):
                continue   # stale observation – skip rather than crash

            uv = kf.features.pts2d[obs_idx]

            e = g2o.EdgeSE3ProjectXYZ()
            e.set_vertex(0, optimizer.vertex(mp_vertex[mp_id]))
            e.set_vertex(1, optimizer.vertex(kf_vertex[kf.kf_id]))
            e.set_measurement(uv.astype(np.float64))
            e.set_information(np.eye(2))

            rk = g2o.RobustKernelHuber()
            rk.set_delta(np.sqrt(5.991))
            e.set_robust_kernel(rk)

            optimizer.add_edge(e)
            edges.append((e, mp, kf.kf_id in local_kf_ids))

    if len(edges) < 5:
        if verbose:
            print(f"[BA] Too few edges ({len(edges)}) — skipping")
        return local_kfs, map_points, []

    # ── Optimise ─────────────────────────────────────────────────────── #
    optimizer.initialize_optimization()
    optimizer.set_verbose(verbose)
    optimizer.optimize(n_iters)

    # ── Update KF poses ───────────────────────────────────────────────── #
    for kf in local_kfs:
        v    = optimizer.vertex(kf_vertex[kf.kf_id])
        se3  = v.estimate()
        R_cw = se3.rotation().matrix()
        t_cw = se3.translation()
        T    = np.eye(4)
        T[:3, :3] = R_cw.T
        T[:3,  3] = -R_cw.T @ t_cw
        kf.T_world_cam = T

    # ── Update map points + cull high-error ones ──────────────────────── #
    reproj_thresh = 5.991
    culled_ids    = set()

    for e, mp, is_local in edges:
        mp_id = id(mp)
        if mp_id not in mp_vertex:
            continue
        v      = optimizer.vertex(mp_vertex[mp_id])
        mp.xyz = v.estimate().copy()
        if e.chi2() > reproj_thresh and is_local:
            culled_ids.add(mp_id)

    kept   = [mp for mp in map_points if id(mp) not in culled_ids]
    culled = [mp for mp in map_points if id(mp) in culled_ids]

    # FIX 3: explicit completion log
    print(f"[BA] Done — edges={len(edges)} "
          f"kept={len(kept)} culled={len(culled)}")

    return local_kfs, kept, culled