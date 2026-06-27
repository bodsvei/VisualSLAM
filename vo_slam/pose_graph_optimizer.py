"""
pose_graph_optimizer.py  (API-robust version)
----------------------------------------------
Fixes in this revision
----------------------
FIX 1 — g2o API compatibility:
  `LinearSolverCholmodSE3` does not exist in all g2o-python builds.
  Now tries four solver configurations in order and falls back to
  scipy automatically. Run check_g2o.py to see your available names.

FIX 2 — EdgeSE3Expmap compatibility:
  Some builds expose EdgeSE3Expmap, others use EdgeSE3.
  Both are tried with automatic detection.

FIX 3 — Full scipy fallback:
  If g2o is present but no compatible solver is found, scipy
  L-BFGS-B is used instead. Slower but always works.
"""

from __future__ import annotations
import numpy as np
import threading
import traceback
from dataclasses import dataclass
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────── #
#  g2o API probe — run once at import time                                #
# ─────────────────────────────────────────────────────────────────────── #
G2O_AVAILABLE = False
G2O_SOLVER    = None    # will hold a lambda that builds the optimizer
G2O_EDGE_TYPE = None    # 'SE3Expmap' | 'SE3'

def _probe_g2o():
    global G2O_AVAILABLE, G2O_SOLVER, G2O_EDGE_TYPE
    try:
        import g2o
    except ImportError:
        return

    # Try solver configurations in order of preference
    solver_attempts = [
        # (description, factory_lambda)
        ("Cholmod SE3",  lambda: g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())),
        ("Eigen SE3",    lambda: g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())),
        ("Dense SE3",    lambda: g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())),
        ("Cholmod X",    lambda: g2o.BlockSolverX(g2o.LinearSolverCholmodX())),
        ("Eigen X",      lambda: g2o.BlockSolverX(g2o.LinearSolverEigenX())),
        ("Dense X",      lambda: g2o.BlockSolverX(g2o.LinearSolverDenseX())),
    ]

    for name, factory in solver_attempts:
        try:
            solver = factory()
            opt    = g2o.SparseOptimizer()
            opt.set_algorithm(g2o.OptimizationAlgorithmLevenberg(solver))
            G2O_SOLVER = factory
            G2O_AVAILABLE = True
            print(f"[PGO] g2o solver: {name}")
            break
        except (AttributeError, Exception):
            continue

    if not G2O_AVAILABLE:
        print("[PGO] g2o present but no compatible solver found — using scipy")
        return

    # Probe edge type
    try:
        _ = g2o.EdgeSE3Expmap
        G2O_EDGE_TYPE = 'SE3Expmap'
    except AttributeError:
        try:
            _ = g2o.EdgeSE3
            G2O_EDGE_TYPE = 'SE3'
        except AttributeError:
            G2O_AVAILABLE = False
            print("[PGO] g2o: no compatible edge type found — using scipy")

_probe_g2o()


@dataclass
class PGOResult:
    success      : bool
    n_poses      : int
    n_loop_edges : int
    iterations   : int


class PoseGraphOptimizer:
    """
    Triggered by verified loop closure events.
    Corrects all keyframe poses and updates the pose graph.

    Parameters
    ----------
    vo       : VisualOdometry instance (poses modified in-place)
    n_iters  : PGO iterations per call
    verbose  : print optimization stats
    """

    def __init__(self, vo, n_iters: int = 20, verbose: bool = True):
        self.vo      = vo
        self.n_iters = n_iters
        self.verbose = verbose
        self._lock   = threading.Lock()

        self.n_optimizations = 0
        self.loop_edges: List[dict] = []

        backend = "g2o" if G2O_AVAILABLE else "scipy"
        print(f"[PGO] backend={backend}  "
              f"edge={G2O_EDGE_TYPE if G2O_AVAILABLE else 'n/a'}")

    # ------------------------------------------------------------------ #
    #  Loop closure callback                                               #
    # ------------------------------------------------------------------ #

    def on_loop(self, event) -> None:
        self.loop_edges.append({
            "query_kf_id" : event.query_kf_id,
            "match_kf_id" : event.match_kf_id,
            "T_rel"       : event.T_rel.copy(),
            "information" : np.eye(6) * float(event.geo_inliers),
        })

        print(f"\n[PGO] Loop constraint: "
              f"KF{event.query_kf_id} ↔ KF{event.match_kf_id}  "
              f"inliers={event.geo_inliers}")

        result = self._optimize()

        if result.success:
            print(f"[PGO] ✓ Optimized {result.n_poses} poses  "
                  f"{result.n_loop_edges} loop edges  "
                  f"({result.iterations} iters)\n")
        else:
            print(f"[PGO] ✗ Skipped (poses={result.n_poses}, "
                  f"loop_edges={result.n_loop_edges})\n")

    # ------------------------------------------------------------------ #
    #  Core dispatch                                                       #
    # ------------------------------------------------------------------ #

    def _optimize(self) -> PGOResult:
        with self._lock:
            kfs = self.vo.keyframes
            if self.verbose:
                print(f"[PGO] Triggered: n_kfs={len(kfs)}, n_loop_edges={len(self.loop_edges)}")
            
            if len(kfs) < 3:
                return PGOResult(False, len(kfs), 0, 0)
            
            if not self.loop_edges:
                return PGOResult(False, len(kfs), 0, 0)

            if G2O_AVAILABLE:
                try:
                    return self._optimize_g2o(kfs)
                except Exception:
                    print("[PGO] g2o optimization failed — falling back to scipy")
                    traceback.print_exc()
            return self._optimize_scipy(kfs)

    # ------------------------------------------------------------------ #
    #  g2o PGO                                                            #
    # ------------------------------------------------------------------ #

    def _create_edge(self, g2o):
        """Creates edge based on probed compatibility [Fix 2]"""
        if G2O_EDGE_TYPE == "SE3Expmap":
            return g2o.EdgeSE3Expmap()
        return g2o.EdgeSE3()

    def _optimize_g2o(self, kfs) -> PGOResult:
        import g2o

        optimizer = g2o.SparseOptimizer()
        optimizer.set_algorithm(
            g2o.OptimizationAlgorithmLevenberg(G2O_SOLVER())
        )

        kf_id_set = {kf.kf_id for kf in kfs}

        # ── Pose vertices ────────────────────────────────────────────── #
        # Safe root selection instead of hardcoded KF 0 [Fix 7]
        root_id = min(k.kf_id for k in kfs)
        for kf in kfs:
            v = g2o.VertexSE3Expmap()
            v.set_id(kf.kf_id)
            R    = kf.T_world_cam[:3, :3]
            t    = kf.T_world_cam[:3,  3]
            v.set_estimate(g2o.SE3Quat(R.T, -R.T @ t))
            v.set_fixed(kf.kf_id == root_id)
            
            # Diagnostic check on vertex insertion [Fix 3]
            ok = optimizer.add_vertex(v)
            if not ok:
                print(f"[PGO] FAILED TO ADD KF {kf.kf_id}")

        # Post-insertion validation diagnostics [Fix 4 & 8]
        for kf in kfs:
            if optimizer.vertex(kf.kf_id) is None:
                print(f"[PGO] MISSING AFTER INSERT KF {kf.kf_id}")

        print(
            f"[PGO] KFs={len(kfs)} "
            f"min={min(k.kf_id for k in kfs)} "
            f"max={max(k.kf_id for k in kfs)}"
        )

        print("  [DEBUG PGO] Adding odometry edges...")
        # ── Sequential odometry edges ────────────────────────────────── #
        for i in range(1, len(kfs)):
            prev_kf = kfs[i - 1]
            curr_kf = kfs[i]
            
            # Safe vertex retrieval to guard against segmentation faults [Fix 5]
            v0 = optimizer.vertex(prev_kf.kf_id)
            v1 = optimizer.vertex(curr_kf.kf_id)

            if v0 is None or v1 is None:
                print(
                    f"[PGO] Missing odom vertex "
                    f"{prev_kf.kf_id}->{curr_kf.kf_id}"
                )
                continue

            T_prev  = self._T_cam_world(prev_kf.T_world_cam)
            T_curr  = self._T_cam_world(curr_kf.T_world_cam)
            T_rel   = T_curr @ np.linalg.inv(T_prev)

            e = self._create_edge(g2o) # Use correct probed edge [Fix 2]
            e.set_vertex(0, v0)
            e.set_vertex(1, v1)
            e.set_measurement(g2o.SE3Quat(T_rel[:3,:3], T_rel[:3,3]))
            # e.set_information(np.eye(6) * 0.5)  # Disabled: Causes segfault in g2o-python bindings
            optimizer.add_edge(e)

        print("  [DEBUG PGO] Adding loop edges...")
        # ── Loop closure edges ───────────────────────────────────────── #
        n_added = 0
        for lc in self.loop_edges:
            qid = lc["query_kf_id"]
            mid = lc["match_kf_id"]
            if qid not in kf_id_set or mid not in kf_id_set:
                continue

            # Safe vertex lookup for loop closure [Fix 5]
            v0 = optimizer.vertex(mid)
            v1 = optimizer.vertex(qid)
            if v0 is None or v1 is None:
                print(f"[PGO] Missing loop vertex {mid}->{qid}")
                continue

            # Bug 3: Use the actual measurement from loop_detector.
            # lc["T_rel"] is already T_{query_cam <- match_cam}, which is exactly what g2o expects.
            T_meas = lc["T_rel"]
            e = self._create_edge(g2o) # Use correct probed edge [Fix 2]
            e.set_vertex(0, v0)
            e.set_vertex(1, v1)
            e.set_measurement(g2o.SE3Quat(T_meas[:3,:3], T_meas[:3,3]))
            # e.set_information(lc["information"]) # Disabled: Causes segfault in g2o-python bindings

            # Robust kernel binding safety wrapper [Fix 6]
            try:
                rk = g2o.RobustKernelHuber()
                rk.set_delta(np.sqrt(5.991))
                e.set_robust_kernel(rk)
            except AttributeError:
                pass

            optimizer.add_edge(e)
            n_added += 1

        if n_added == 0:
            return PGOResult(False, len(kfs), 0, 0)

        print("  [DEBUG PGO] Initializing optimization...")
        optimizer.initialize_optimization()
        print("  [DEBUG PGO] Optimizing...")
        optimizer.set_verbose(self.verbose)
        optimizer.optimize(self.n_iters)
        self.n_optimizations += 1

        # ── Write back corrected poses ───────────────────────────────── #
        if not kfs:
            return PGOResult(True, 0, n_added, self.n_iters)
            
        old_latest_T = kfs[-1].T_world_cam.copy()
        old_poses = {kf.kf_id: kf.T_world_cam.copy() for kf in kfs}
        
        with self.vo.lock:
            for kf in kfs:
                v    = optimizer.vertex(kf.kf_id)
                if v is None:
                    continue

                se3  = v.estimate()
                R_cw = se3.rotation().matrix()
                t_cw = se3.translation()
                
                # Bug 2 check: T_world_cam = [R_cw.T | -R_cw.T @ t_cw]
                T = np.eye(4)
                T[:3,:3] = R_cw.T
                T[:3, 3] = -R_cw.T @ t_cw
                kf.T_world_cam = T

            # Update current tracking pose to maintain relative continuity
            new_latest_T = kfs[-1].T_world_cam
            correction = new_latest_T @ np.linalg.inv(old_latest_T)
            self.vo.T_world_cam = correction @ self.vo.T_world_cam

            self._update_map_and_graph(kfs, old_poses)

        return PGOResult(True, len(kfs), n_added, self.n_iters)

    def _update_map_and_graph(self, kfs: List[Keyframe], old_poses: dict):
        """
        Update all map points and non-keyframe trajectory poses after optimization.
        """
        # 1. Build a map of kf_id -> correction matrix
        # Delta = T_new_world_kf @ inv(T_old_world_kf)
        corrections = {}
        for kf in kfs:
            T_old = old_poses.get(kf.kf_id)
            if T_old is not None:
                corrections[kf.kf_id] = kf.T_world_cam @ np.linalg.inv(T_old)

        if not corrections:
            return

        # 2. Update map points by "dragging" them with their reference keyframe
        for mp in self.vo.map_points:
            if not mp.obs: continue
            ref_kf_id = min(mp.obs.keys())
            corr = corrections.get(ref_kf_id)
            if corr is not None:
                p_hom = np.append(mp.xyz, 1.0)
                mp.xyz = (corr @ p_hom)[:3]

        # 3. Update all poses in the pose graph (keyframes and non-keyframes)
        self.vo.pose_graph.transform_all(kfs, corrections)

    # ------------------------------------------------------------------ #
    #  scipy PGO fallback                                                  #
    # ------------------------------------------------------------------ #

    def _optimize_scipy(self, kfs) -> PGOResult:
        from scipy.optimize import minimize
        import cv2

        n             = len(kfs)
        kf_id_to_idx  = {kf.kf_id: i for i, kf in enumerate(kfs)}
        kf_id_set     = set(kf_id_to_idx.keys())

        def pack(kfs):
            x = np.zeros(n * 6)
            for i, kf in enumerate(kfs):
                R   = kf.T_world_cam[:3,:3]
                t   = kf.T_world_cam[:3, 3]
                rv, _= cv2.Rodrigues(R)
                x[i*6:i*6+3]  = rv.ravel()
                x[i*6+3:i*6+6]= t
            return x

        def unpack(x, i):
            rv = x[i*6:i*6+3]
            t  = x[i*6+3:i*6+6]
            R, _= cv2.Rodrigues(rv)
            T   = np.eye(4)
            T[:3,:3] = R
            T[:3, 3] = t
            return T

        x0 = pack(kfs)

        # Odometry constraints
        odom = []
        for i in range(1, n):
            T_rel = np.linalg.inv(kfs[i-1].T_world_cam) @ kfs[i].T_world_cam
            odom.append((i-1, i, T_rel, 0.5))

        # Loop constraints
        loops = []
        for lc in self.loop_edges:
            qid = lc["query_kf_id"]
            mid = lc["match_kf_id"]
            if qid not in kf_id_set or mid not in kf_id_set:
                continue
            w = float(np.trace(lc["information"])) / 6.0
            loops.append((kf_id_to_idx[mid], kf_id_to_idx[qid], lc["T_rel"], w))

        if not loops:
            return PGOResult(False, n, 0, 0)

        def cost(x):
            from scipy.spatial.transform import Rotation
            total = 0.0
            for (i, j, T_ij, w) in odom + loops:
                Ti    = unpack(x, i)
                Tj    = unpack(x, j)
                T_est = np.linalg.inv(Ti) @ Tj

                # Bug 4: Correct rotation residual using SO(3) logarithmic map
                err_R = Rotation.from_matrix(T_est[:3,:3].T @ T_ij[:3,:3]).as_rotvec()

                err_t = T_est[:3, 3] - T_ij[:3, 3]
                total += w * (np.sum(err_R**2) + np.sum(err_t**2))
            return total

        # Fix first pose
        bounds = [(None,None)] * (n * 6)
        for j in range(6):
            bounds[j] = (x0[j], x0[j])

        res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': self.n_iters * 20, 'ftol': 1e-8})

        x_opt = res.x
        if not kfs:
             return PGOResult(res.success, 0, len(loops), self.n_iters)
             
        old_latest_T = kfs[-1].T_world_cam.copy()
        old_poses = {kf.kf_id: kf.T_world_cam.copy() for kf in kfs}

        for i, kf in enumerate(kfs):
            T = unpack(x_opt, i)
            kf.T_world_cam = T

        # Update current tracking pose to maintain relative continuity
        new_latest_T = kfs[-1].T_world_cam
        correction = new_latest_T @ np.linalg.inv(old_latest_T)
        self.vo.T_world_cam = correction @ self.vo.T_world_cam
        self._update_map_and_graph(kfs, old_poses)

        self.n_optimizations += 1
        return PGOResult(res.success, n, len(loops), self.n_iters)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _T_cam_world(T_world_cam):
        R = T_world_cam[:3,:3]
        t = T_world_cam[:3, 3]
        T = np.eye(4)
        T[:3,:3] = R.T
        T[:3, 3] = -R.T @ t
        return T

    def summary(self) -> str:
        """Safe summary reporting to prevent runtime keyframe crashes [Fix 1]"""
        kfs = self.vo.keyframes

        if not kfs:
            return (
                f"PoseGraphOptimizer | "
                f"backend={'g2o' if G2O_AVAILABLE else 'scipy'} | "
                f"optimizations={self.n_optimizations} | "
                f"loop_edges={len(self.loop_edges)}"
            )

        return (
            f"PoseGraphOptimizer | "
            f"backend={'g2o' if G2O_AVAILABLE else 'scipy'} | "
            f"optimizations={self.n_optimizations} | "
            f"loop_edges={len(self.loop_edges)} | "
            f"minKF={min(k.kf_id for k in kfs)} | "
            f"maxKF={max(k.kf_id for k in kfs)} | "
            f"count={len(kfs)}"
        )