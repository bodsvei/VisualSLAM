"""
bundle_adjustment.py
---------------------
Robust against multiple g2o edge APIs.

The installed g2o build may expose EdgeSE3ProjectXYZ without .fx/.fy/.cx/.cy
attributes. This version probes the available API once at startup and
selects the best available strategy:

  Strategy A — direct attributes (e.fx / e.fy / e.cx / e.cy)
    Standard g2opy build (pip install g2o-python on most systems)

  Strategy B — EdgeSE3ProjectXYZOnlyPose (pose-only BA)
    When the joint projection edge lacks intrinsic setters.
    Optimises poses while holding 3-D points fixed.
    Still far better than no BA — corrects rotational/translational drift.

  Strategy C — scipy joint BA (full fallback)
    When g2o projection edges are completely unavailable.
    Slower but optimises both poses AND points.

Run  python3 check_g2o.py  to see which strategy will be selected.
"""

from __future__ import annotations
import numpy as np
import traceback
from typing import List, Set, Tuple
from .keyframe      import Keyframe
from .triangulation import MapPoint
from .camera        import CameraModel

try:
    import g2o
    G2O_AVAILABLE = True
except ImportError:
    g2o = None
    G2O_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────── #
#  One-time API probe                                                      #
# ─────────────────────────────────────────────────────────────────────── #

_BA_STRATEGY      = None    # 'joint' | 'pose_only' | 'scipy'
_BA_SOLVER_FACTORY = None

def _probe():
    global _BA_STRATEGY, _BA_SOLVER_FACTORY

    if not G2O_AVAILABLE:
        _BA_STRATEGY = 'scipy'
        return

    # ── Solver probe ─────────────────────────────────────────────────── #
    for name, factory in [
        ("Eigen SE3",   lambda: g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())),
        ("Dense SE3",   lambda: g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())),
        ("Cholmod SE3", lambda: g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())),
        ("Eigen X",     lambda: g2o.BlockSolverX(g2o.LinearSolverEigenX())),
        ("Dense X",     lambda: g2o.BlockSolverX(g2o.LinearSolverDenseX())),
    ]:
        try:
            factory()
            _BA_SOLVER_FACTORY = factory
            print(f"[BA] Solver: {name}")
            break
        except (AttributeError, Exception):
            continue

    if _BA_SOLVER_FACTORY is None:
        _BA_STRATEGY = 'scipy'
        print("[BA] No g2o solver found — using scipy")
        return

    # ── Edge probe ───────────────────────────────────────────────────── #
    # Strategy A: joint BA with direct-attribute intrinsics
    try:
        e = g2o.EdgeSE3ProjectXYZ()
        e.fx = 1.0   # test if writable
        e.fy = 1.0
        e.cx = 0.0
        e.cy = 0.0
        _BA_STRATEGY = 'joint'
        print("[BA] Edge API: EdgeSE3ProjectXYZ (joint pose+point BA)")
        return
    except (AttributeError, Exception):
        pass

    # Strategy B: pose-only BA
    try:
        e = g2o.EdgeSE3ProjectXYZOnlyPose()
        e.fx = 1.0
        _BA_STRATEGY = 'pose_only'
        print("[BA] Edge API: EdgeSE3ProjectXYZOnlyPose (pose-only BA)")
        return
    except (AttributeError, Exception):
        pass

    # Strategy C: scipy fallback
    _BA_STRATEGY = 'scipy'
    print("[BA] g2o projection edges unavailable — using scipy joint BA")

_probe()


# ─────────────────────────────────────────────────────────────────────── #
#  Public entry point                                                      #
# ─────────────────────────────────────────────────────────────────────── #

def local_bundle_adjustment(
    local_kfs  : List[Keyframe],
    fixed_kfs  : List[Keyframe],
    map_points : List[MapPoint],
    camera     : CameraModel,
    n_iters    : int  = 10,
    verbose    : bool = False,
) -> Tuple[List[Keyframe], List[MapPoint], List[MapPoint], bool]:

    if len(local_kfs) < 2 or len(map_points) < 5:
        return local_kfs, map_points, [], False

    if _BA_STRATEGY == 'joint':
        kfs, kept, culled = _ba_joint_g2o(local_kfs, fixed_kfs, map_points, camera, n_iters, verbose)
        return kfs, kept, culled, True
    elif _BA_STRATEGY == 'pose_only':
        kfs, kept, culled = _ba_pose_only_g2o(local_kfs, fixed_kfs, map_points, camera, n_iters, verbose)
        return kfs, kept, culled, True
    else:
        return _ba_scipy(local_kfs, fixed_kfs, map_points, camera, n_iters, verbose)


# ─────────────────────────────────────────────────────────────────────── #
#  Strategy A — Joint pose + point BA (g2o, EdgeSE3ProjectXYZ)            #
# ─────────────────────────────────────────────────────────────────────── #

def _ba_joint_g2o(local_kfs, fixed_kfs, map_points, camera, n_iters, verbose):
    optimizer = _make_optimizer()
    next_id   = 0
    all_kfs   = local_kfs + fixed_kfs
    anchor_id = min(kf.kf_id for kf in all_kfs)

    kf_vertex: dict = {}
    for kf in all_kfs:
        v = g2o.VertexSE3Expmap()
        v.set_id(next_id)
        R, t = kf.T_world_cam[:3,:3], kf.T_world_cam[:3,3]
        v.set_estimate(g2o.SE3Quat(R.T, -R.T @ t))
        v.set_fixed(kf.kf_id == anchor_id or kf in fixed_kfs or kf.kf_id == 0)
        optimizer.add_vertex(v)
        kf_vertex[kf.kf_id] = next_id
        next_id += 1

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

    local_kf_ids: Set[int] = {kf.kf_id for kf in local_kfs}
    edges = []

    for kf in all_kfs:
        if kf.kf_id not in kf_vertex:
            continue
        for mp in kf.map_points:
            mp_id = id(mp)
            if mp_id not in mp_vertex:
                continue
            obs_idx = mp.obs.get(kf.kf_id)
            if obs_idx is None or obs_idx >= len(kf.features.pts2d):
                continue
            uv = kf.features.pts2d[obs_idx]

            e = g2o.EdgeSE3ProjectXYZ()
            e.set_vertex(0, optimizer.vertex(mp_vertex[mp_id]))
            e.set_vertex(1, optimizer.vertex(kf_vertex[kf.kf_id]))
            e.set_measurement(uv.astype(np.float64))
            e.set_information(np.eye(2))
            e.fx, e.fy, e.cx, e.cy = camera.fx, camera.fy, camera.cx, camera.cy
            rk = g2o.RobustKernelHuber(); rk.set_delta(np.sqrt(5.991))
            e.set_robust_kernel(rk)
            optimizer.add_edge(e)
            edges.append((e, mp, kf.kf_id in local_kf_ids))

    if len(edges) < 5:
        return local_kfs, map_points, []

    optimizer.initialize_optimization()
    optimizer.set_verbose(verbose)
    optimizer.optimize(n_iters)

    for kf in local_kfs:
        v = optimizer.vertex(kf_vertex[kf.kf_id]); se3 = v.estimate()
        R_cw = se3.rotation().matrix(); t_cw = se3.translation()
        T = np.eye(4); T[:3,:3] = R_cw.T; T[:3,3] = -R_cw.T @ t_cw
        kf.T_world_cam = T

    culled_ids = set()
    for e, mp, is_local in edges:
        v = optimizer.vertex(mp_vertex[id(mp)])
        mp.xyz = v.estimate().copy()
        if e.chi2() > 5.991 and is_local:
            culled_ids.add(id(mp))

    kept   = [mp for mp in map_points if id(mp) not in culled_ids]
    culled = [mp for mp in map_points if id(mp) in culled_ids]
    print(f"[BA] joint g2o — edges={len(edges)} kept={len(kept)} culled={len(culled)}")
    return local_kfs, kept, culled


# ─────────────────────────────────────────────────────────────────────── #
#  Strategy B — Pose-only BA (g2o, EdgeSE3ProjectXYZOnlyPose)            #
# ─────────────────────────────────────────────────────────────────────── #

def _ba_pose_only_g2o(local_kfs, fixed_kfs, map_points, camera, n_iters, verbose):
    """
    Optimise only camera poses; 3-D points are held fixed as measurements.
    Less powerful than joint BA but still corrects inter-KF pose drift.
    """
    optimizer = _make_optimizer()
    next_id   = 0
    all_kfs   = local_kfs + fixed_kfs
    anchor_id = min(kf.kf_id for kf in all_kfs)

    kf_vertex: dict = {}
    for kf in all_kfs:
        v = g2o.VertexSE3Expmap()
        v.set_id(next_id)
        R, t = kf.T_world_cam[:3,:3], kf.T_world_cam[:3,3]
        v.set_estimate(g2o.SE3Quat(R.T, -R.T @ t))
        v.set_fixed(kf.kf_id == anchor_id or kf in fixed_kfs or kf.kf_id == 0)
        optimizer.add_vertex(v)
        kf_vertex[kf.kf_id] = next_id
        next_id += 1

    edges_added = 0
    for kf in all_kfs:
        if kf.kf_id not in kf_vertex:
            continue
        for mp in kf.map_points:
            obs_idx = mp.obs.get(kf.kf_id)
            if obs_idx is None or obs_idx >= len(kf.features.pts2d):
                continue
            uv = kf.features.pts2d[obs_idx]

            e = g2o.EdgeSE3ProjectXYZOnlyPose()
            e.set_vertex(0, optimizer.vertex(kf_vertex[kf.kf_id]))
            e.set_measurement(uv.astype(np.float64))
            e.set_information(np.eye(2))
            e.fx, e.fy, e.cx, e.cy = camera.fx, camera.fy, camera.cx, camera.cy
            e.Xw = mp.xyz.astype(np.float64)   # fixed 3-D point
            rk = g2o.RobustKernelHuber(); rk.set_delta(np.sqrt(5.991))
            e.set_robust_kernel(rk)
            optimizer.add_edge(e)
            edges_added += 1

    if edges_added < 5:
        return local_kfs, map_points, []

    optimizer.initialize_optimization()
    optimizer.set_verbose(verbose)
    optimizer.optimize(n_iters)

    for kf in local_kfs:
        v = optimizer.vertex(kf_vertex[kf.kf_id]); se3 = v.estimate()
        R_cw = se3.rotation().matrix(); t_cw = se3.translation()
        T = np.eye(4); T[:3,:3] = R_cw.T; T[:3,3] = -R_cw.T @ t_cw
        kf.T_world_cam = T

    print(f"[BA] pose-only g2o — edges={edges_added} "
          f"KFs optimised={len(local_kfs)}")
    return local_kfs, map_points, []   # points not culled in pose-only mode


# ─────────────────────────────────────────────────────────────────────── #
#  Strategy C — scipy joint BA (full fallback)                            #
# ─────────────────────────────────────────────────────────────────────── #

def _ba_scipy(local_kfs, fixed_kfs, map_points, camera, n_iters, verbose):
    """
    Joint pose + point optimisation using scipy L-BFGS-B.
    Slower than g2o but works on any Python installation.
    """
    try:
        from scipy.optimize import minimize
        import cv2 as _cv2
    except ImportError:
        print("[BA] scipy not available — skipping")
        return local_kfs, map_points, []

    all_kfs   = local_kfs + fixed_kfs
    anchor_id = min(kf.kf_id for kf in all_kfs)
    kf_ids    = [kf.kf_id for kf in local_kfs]
    fixed_ids = {kf.kf_id for kf in fixed_kfs} | {anchor_id}

    # Pack: 6 DoF per free KF + 3 DoF per MP
    free_kfs   = [kf for kf in local_kfs if kf.kf_id not in fixed_ids]
    free_kf_idx = {kf.kf_id: i for i, kf in enumerate(free_kfs)}
    n_kf = len(free_kfs)
    n_mp = len(map_points)
    mp_idx = {id(mp): i for i, mp in enumerate(map_points)}

    def pack(free_kfs, mps):
        x = np.zeros(n_kf * 6 + n_mp * 3)
        for i, kf in enumerate(free_kfs):
            R = kf.T_world_cam[:3,:3]; t = kf.T_world_cam[:3,3]
            R_cw = R.T
            t_cw = -R.T @ t
            rv, _ = _cv2.Rodrigues(R_cw)
            x[i*6:i*6+3] = rv.ravel()
            x[i*6+3:i*6+6] = t_cw.ravel()
        for i, mp in enumerate(mps):
            x[n_kf*6 + i*3: n_kf*6 + i*3+3] = mp.xyz
        return x

    def unpack_kf(x, i):
        rv = x[i*6:i*6+3]; t_cw = x[i*6+3:i*6+6]
        R_cw, _ = _cv2.Rodrigues(rv)
        R_wc = R_cw.T
        t_wc = -R_cw.T @ t_cw.reshape(-1, 1)
        T = np.eye(4)
        T[:3,:3] = R_wc
        T[:3,3] = t_wc.ravel()
        return T

    def unpack_mp(x, i):
        return x[n_kf*6 + i*3: n_kf*6 + i*3+3]

    K = camera.K
    observations = []  # (kf_idx_in_free, mp_idx, uv)
    for kf in free_kfs:
        for mp in kf.map_points:
            obs_idx = mp.obs.get(kf.kf_id)
            if obs_idx is None or obs_idx >= len(kf.features.pts2d): continue
            mpi = mp_idx.get(id(mp))
            if mpi is None: continue
            observations.append((free_kf_idx[kf.kf_id], mpi, kf.features.pts2d[obs_idx]))

    if len(observations) < 5:
        return local_kfs, map_points, []

    x0 = pack(free_kfs, map_points)

    # Pre-extract observation data for vectorization
    kf_indices = np.array([obs[0] for obs in observations])
    mp_indices = np.array([obs[1] for obs in observations])
    uv_targets = np.array([obs[2] for obs in observations])

    def cost(x):
        # 1. Extract all poses and points
        kfs_x = x[:n_kf * 6].reshape(n_kf, 6)
        mps_x = x[n_kf * 6:].reshape(n_mp, 3)
        
        # 2. Gather specific poses and points for each observation
        obs_kfs = kfs_x[kf_indices]
        obs_mps = mps_x[mp_indices]
        
        total_err = 0.0
        # Iterate over batched observations (much faster than individual Python loops)
        for i in range(len(observations)):
            rv, t = obs_kfs[i, :3], obs_kfs[i, 3:6]
            xyz = obs_mps[i]
            
            R, _ = _cv2.Rodrigues(rv)
            p = R @ xyz + t
            
            if p[2] < 1e-4:
                total_err += 1e12
                continue
                
            pred_u = K[0,0] * p[0] / p[2] + K[0,2]
            pred_v = K[1,1] * p[1] / p[2] + K[1,2]
            
            du = pred_u - uv_targets[i, 0]
            dv = pred_v - uv_targets[i, 1]
            total_err += float(du*du + dv*dv)
            
        return total_err

    res = minimize(cost, x0, method='L-BFGS-B',
                   options={'maxiter': n_iters * 50, 'ftol': 1e-8})

    x_opt = res.x
    for i, kf in enumerate(free_kfs):
        kf.T_world_cam = unpack_kf(x_opt, i)
    for i, mp in enumerate(map_points):
        mp.xyz = unpack_mp(x_opt, i).copy()

    print(f"[BA] scipy joint — obs={len(observations)} "
          f"KFs={len(free_kfs)} MPs={n_mp} "
          f"converged={res.success}")
    return local_kfs, map_points, []


# ─────────────────────────────────────────────────────────────────────── #
#  Helpers                                                                 #
# ─────────────────────────────────────────────────────────────────────── #

def _make_optimizer():
    optimizer = g2o.SparseOptimizer()
    optimizer.set_algorithm(
        g2o.OptimizationAlgorithmLevenberg(_BA_SOLVER_FACTORY())
    )
    return optimizer