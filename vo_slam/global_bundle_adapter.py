"""
global_bundle_adjustment.py
Full BA after loop closure / at end of sequence.
"""

import numpy as np
from typing import List
from .keyframe      import Keyframe
from .triangulation import MapPoint
from .camera        import CameraModel
from .bundle_adjustment import _build_optimizer, G2O_AVAILABLE

def global_bundle_adjustment(
    all_keyframes : List[Keyframe],
    all_map_points: List[MapPoint],
    camera        : CameraModel,
    n_iters       : int = 50,
    verbose       : bool = False,
) -> None:
    if not G2O_AVAILABLE:
        print("[GlobalBA] g2o not available – skipping")
        return
    if len(all_keyframes) < 2 or len(all_map_points) < 5:
        return

    optimizer = _build_optimizer()
    next_id = 0

    # Camera parameter
    cam_param = g2o.ParameterCamera()
    cam_param.setKcam(camera.fx, camera.fy, camera.cx, camera.cy)
    cam_param.set_id(0)
    optimizer.add_parameter(cam_param)

    # Anchor: first keyframe (lowest kf_id)
    anchor_id = min(kf.kf_id for kf in all_keyframes)

    # Pose vertices
    kf_vertex = {}
    for kf in all_keyframes:
        v = g2o.VertexSE3Expmap()
        v.set_id(next_id)
        R = kf.T_world_cam[:3, :3]
        t = kf.T_world_cam[:3, 3]
        R_cw = R.T
        t_cw = -R.T @ t
        v.set_estimate(g2o.SE3Quat(R_cw, t_cw))
        v.set_fixed(kf.kf_id == anchor_id)
        optimizer.add_vertex(v)
        kf_vertex[kf.kf_id] = next_id
        next_id += 1

    # Map point vertices
    mp_vertex = {}
    for mp in all_map_points:
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

    # Edges
    edges = []
    for kf in all_keyframes:
        if kf.kf_id not in kf_vertex:
            continue
        for mp in kf.map_points:
            mp_id = id(mp)
            if mp_id not in mp_vertex:
                continue
            if kf.kf_id in mp.obs:
                obs_idx = mp.obs[kf.kf_id]
            else:
                obs_idx = mp.ref_idx if kf.kf_id == 0 else mp.cur_idx
            if obs_idx is None or obs_idx >= len(kf.features.pts2d):
                continue
            uv = kf.features.pts2d[obs_idx]
            e = g2o.EdgeSE3ProjectXYZ()
            e.set_vertex(0, optimizer.vertex(mp_vertex[mp_id]))
            e.set_vertex(1, optimizer.vertex(kf_vertex[kf.kf_id]))
            e.set_measurement(uv.astype(np.float64))
            e.set_information(np.eye(2))
            rk = g2o.RobustKernelHuber()
            rk.set_delta(np.sqrt(5.991))
            e.set_robust_kernel(rk)
            e.set_parameter_id(0, 0)
            optimizer.add_edge(e)
            edges.append(e)

    if len(edges) < 5:
        if verbose:
            print("[GlobalBA] Too few edges – skipping")
        return

    optimizer.initialize_optimization()
    optimizer.set_verbose(verbose)
    optimizer.optimize(n_iters)

    # Update keyframe poses
    for kf in all_keyframes:
        v = optimizer.vertex(kf_vertex[kf.kf_id])
        se3 = v.estimate()
        R_cw = se3.rotation().matrix()
        t_cw = se3.translation()
        T = np.eye(4)
        T[:3, :3] = R_cw.T
        T[:3, 3] = -R_cw.T @ t_cw
        kf.T_world_cam = T

    # Update map points
    for mp in all_map_points:
        mp_id = id(mp)
        if mp_id in mp_vertex:
            v = optimizer.vertex(mp_vertex[mp_id])
            mp.xyz = v.estimate().copy()

    if verbose:
        print(f"[GlobalBA] Done: KFs={len(all_keyframes)} MPs={len(all_map_points)}")