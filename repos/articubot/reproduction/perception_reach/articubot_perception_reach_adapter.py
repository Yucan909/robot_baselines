"""Perception-conditioned ArticuBot reach helpers.

The adapter consumes only the frozen benchmark camera products.  The requested
link identity selects a per-pixel SAPIEN link-instance mask; no handle position,
joint axis, joint type, or motion direction is read.  ArticuBot inference is
performed in the Panda-root frame used by the released RoboGen policy, while
execution remains in the frozen SAPIEN world frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from articubot_observation_adapter import sample_point_cloud


def _actor_id(actor) -> int:
    return int(actor.get_id() if hasattr(actor, "get_id") else actor.id)


def _root_transform(panda):
    T = np.asarray(panda.robot.get_root_pose().to_transformation_matrix(), dtype=np.float64)
    return T[:3, :3], T[:3, 3]


def world_to_root(points, R_world_root, p_world_root):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (points - p_world_root[None, :]) @ R_world_root


def root_to_world(points, R_world_root, p_world_root):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ R_world_root.T + p_world_root[None, :]


def _observation_from_points(points_root, panda, R_world_root, p_world_root, num_points, rng):
    point_cloud = sample_point_cloud(points_root, int(num_points), rng)
    T_world = np.asarray(panda.get_grasp_pose_matrix(), dtype=np.float64)
    p_root = (T_world[:3, 3] - p_world_root) @ R_world_root
    R_root = R_world_root.T @ T_world[:3, :3]
    orient_6d = np.asarray([R_root[:, 0], R_root[:, 1]], dtype=np.float64).reshape(6)
    finger = float(np.asarray(panda.get_finger_qpos(), dtype=np.float64).mean())
    agent_pos = np.concatenate([p_root, orient_6d, [finger]]).astype(np.float32)

    gripper_world = np.asarray(
        [
            panda.hand_link.get_pose().p,
            panda.right_finger_link.get_pose().p,
            panda.left_finger_link.get_pose().p,
            panda.grasp_link.get_pose().p,
        ],
        dtype=np.float64,
    ).reshape(4, 3)
    gripper_pcd = world_to_root(gripper_world, R_world_root, p_world_root).astype(np.float32)
    nearest = point_cloud[np.argmin(cdist(gripper_pcd, point_cloud), axis=1)]
    displacement = (nearest - gripper_pcd).astype(np.float32)
    return {
        "point_cloud": point_cloud,
        "agent_pos": agent_pos,
        "gripper_pcd": gripper_pcd,
        "displacement_gripper_to_object": displacement,
    }


def capture_link_conditioned_observations(*, scene, camera, obj, target_link, panda,
                                           num_points: int, rng: np.random.Generator):
    """Return full-object and target-link observations from one frozen-camera image."""
    scene.update_render()
    camera.take_picture()
    position = np.asarray(camera.get_float_texture("Position"))
    segmentation = np.asarray(camera.get_uint32_texture("Segmentation"))
    actor_seg = segmentation[..., 1].astype(np.int64)
    valid = position[..., 3] > 0
    all_ids = np.asarray([_actor_id(link) for link in obj.get_links()], dtype=np.int64)
    target_id = _actor_id(target_link)
    full_keep = valid & np.isin(actor_seg, all_ids)
    target_keep = valid & (actor_seg == target_id)
    rest_keep = full_keep & ~target_keep
    model_matrix = np.asarray(camera.get_model_matrix(), dtype=np.float64)

    def to_world(mask):
        pc = position[..., :3][mask].astype(np.float64)
        if len(pc) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return pc @ model_matrix[:3, :3].T + model_matrix[:3, 3]

    full_world = to_world(full_keep)
    target_world = to_world(target_keep)
    rest_world = to_world(rest_keep)
    if len(full_world) < 32:
        raise RuntimeError(f"object_not_visible: {len(full_world)} points")
    if len(target_world) < 32:
        raise RuntimeError(f"target_link_not_visible: {len(target_world)} points")
    R_world_root, p_world_root = _root_transform(panda)
    full_root = world_to_root(full_world, R_world_root, p_world_root)
    target_root = world_to_root(target_world, R_world_root, p_world_root)
    full_observation = _observation_from_points(
        full_root, panda, R_world_root, p_world_root, num_points, rng
    )
    mask_distance, _ = cKDTree(target_root).query(
        np.asarray(full_observation["point_cloud"], dtype=np.float64), k=1
    )
    target_mask_on_full = np.asarray(mask_distance < 1e-5, dtype=bool)
    if int(target_mask_on_full.sum()) < 8:
        # At very small visible links, use the nearest full-cloud samples while
        # keeping the network input unchanged.
        order = np.argsort(mask_distance)
        target_mask_on_full[order[: min(32, len(order))]] = True
    return {
        "full": full_observation,
        "target": _observation_from_points(
            target_root, panda, R_world_root, p_world_root, num_points, rng
        ),
        "full_points_world": full_world,
        "target_points_world": target_world,
        "rest_points_world": rest_world,
        "camera_position_world": model_matrix[:3, 3].copy(),
        "target_pixel_count": int(target_keep.sum()),
        "object_pixel_count": int(full_keep.sum()),
        "target_actor_id": target_id,
        "target_mask_on_full_sample": target_mask_on_full,
        "R_world_root": R_world_root,
        "p_world_root": p_world_root,
    }


@torch.no_grad()
def masked_high_level_infer(policy, history, target_mask):
    """Run the unchanged high-level net, restricting its spatial vote to the link mask."""
    pointcloud = torch.from_numpy(np.asarray(history["point_cloud"][-1])[None]).to(policy.device)
    gripper = torch.from_numpy(np.asarray(history["gripper_pcd"][-1])[None]).to(policy.device)
    inputs = torch.cat([pointcloud, gripper], dim=1)
    outputs = policy.high_level_policy(inputs.permute(0, 2, 1))
    weights = outputs[:, :-4, -1]
    displacements = outputs[:, :-4, :-1].reshape(1, -1, 4, 3)
    goals = displacements + inputs[:, :-4, :3].unsqueeze(2)
    mask = torch.from_numpy(np.asarray(target_mask, dtype=bool).reshape(1, -1)).to(policy.device)
    if mask.shape != weights.shape or int(mask.sum()) < 1:
        raise RuntimeError(f"invalid target mask {tuple(mask.shape)} for {tuple(weights.shape)}")
    restricted = weights.masked_fill(~mask, torch.finfo(weights.dtype).min)
    goal = (goals * torch.softmax(restricted, dim=1)[..., None, None]).sum(dim=1)
    goal = goal.unsqueeze(1)
    if goal.shape != (1, 1, 4, 3) or not torch.isfinite(goal).all():
        raise RuntimeError("masked high-level goal is invalid")
    return goal.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def masked_high_level_goal_modes(policy, history, target_mask, max_modes=4, nms_m=0.05):
    """Return link-masked aggregate plus spatially distinct high-response modes.

    The released network predicts one four-keypoint displacement and one
    weight per scene point.  The official wrapper averages all votes.  Mask
    conditioning makes that average safer, but it can still land inside a
    large door/drawer panel.  This exposes the network's own strongest masked
    modes without changing weights or adding any oracle geometry.
    """
    pointcloud = torch.from_numpy(np.asarray(history["point_cloud"][-1])[None]).to(policy.device)
    gripper = torch.from_numpy(np.asarray(history["gripper_pcd"][-1])[None]).to(policy.device)
    inputs = torch.cat([pointcloud, gripper], dim=1)
    outputs = policy.high_level_policy(inputs.permute(0, 2, 1))
    weights = outputs[:, :-4, -1]
    displacements = outputs[:, :-4, :-1].reshape(1, -1, 4, 3)
    goals = displacements + inputs[:, :-4, :3].unsqueeze(2)
    mask = torch.from_numpy(np.asarray(target_mask, dtype=bool).reshape(1, -1)).to(policy.device)
    if mask.shape != weights.shape or int(mask.sum()) < 1:
        raise RuntimeError(f"invalid target mask {tuple(mask.shape)} for {tuple(weights.shape)}")
    restricted = weights.masked_fill(~mask, torch.finfo(weights.dtype).min)
    aggregate = (
        goals * torch.softmax(restricted, dim=1)[..., None, None]
    ).sum(dim=1)[0]
    order = torch.argsort(restricted[0], descending=True)
    selected = []
    selected_scores = []
    for index in order.detach().cpu().tolist():
        if not bool(mask[0, index]):
            break
        candidate = goals[0, index]
        anchor = candidate[3]
        if any(float(torch.linalg.norm(anchor - prior[3])) < float(nms_m) for prior in selected):
            continue
        selected.append(candidate)
        selected_scores.append(float(restricted[0, index].detach().cpu()))
        if len(selected) >= int(max_modes):
            break
    all_goals = [aggregate] + selected
    stacked = torch.stack(all_goals, dim=0).unsqueeze(1)
    if not torch.isfinite(stacked).all():
        raise RuntimeError("masked high-level goal modes contain NaN/Inf")
    return stacked.detach().cpu().numpy().astype(np.float32), {
        "ordering": "official_masked_softmax_aggregate_then_masked_logit_modes",
        "mode_logits": selected_scores,
        "spatial_nms_m": float(nms_m),
        "num_modes_including_aggregate": int(len(all_goals)),
    }


def goal_root_to_world(goal_root, capture):
    goal = np.asarray(goal_root, dtype=np.float64).reshape(4, 3)
    return root_to_world(goal, capture["R_world_root"], capture["p_world_root"])


def shift_goal_to_visible_free_edge(goal_points, target_points, rest_points):
    """Move a predicted door grasp to its visually exposed free edge.

    The free edge is inferred as the target-mask region farthest from all
    other visible object links.  No joint, hinge, axis, handle label, mesh
    semantic, or articulation direction is used.  The official four-point
    gripper geometry is translated rigidly and its relative prediction is
    otherwise unchanged.
    """
    goal = np.asarray(goal_points, dtype=np.float64).reshape(4, 3).copy()
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    rest = np.asarray(rest_points, dtype=np.float64).reshape(-1, 3)
    if len(target) < 32 or len(rest) < 32:
        return goal, {"applied": False, "reason": "insufficient_visible_points"}
    separation, _ = cKDTree(rest).query(target, k=1)
    cutoff = float(np.quantile(separation, 0.90))
    exposed = target[separation >= cutoff]
    if len(exposed) < 16:
        return goal, {"applied": False, "reason": "insufficient_exposed_edge_points"}
    # The official policy still chooses the height along that free edge.  XY
    # has a lower weight because an interior panel prediction is precisely the
    # failure this perceptual conditioning is meant to repair.
    delta = exposed - goal[3]
    score = np.abs(delta[:, 2]) + 0.20 * np.linalg.norm(delta[:, :2], axis=1)
    anchor = exposed[int(np.argmin(score))]
    shift = anchor - goal[3]
    goal += shift[None, :]
    return goal, {
        "applied": True,
        "selection": "target_to_rest_top10pct_then_official_goal_height",
        "separation_cutoff_m": cutoff,
        "n_exposed_points": int(len(exposed)),
        "anchor_world": anchor.tolist(),
        "goal_translation_m": shift.tolist(),
    }


def shift_goal_to_visible_protrusion(goal_points, target_points, camera_position):
    """Translate an official goal onto a protrusion detected in the link mask.

    A dominant plane is estimated from visible target-link points.  Only a
    sufficiently large camera-facing residual is accepted, so a plain panel
    does not invent a handle.  This uses neither mesh labels nor joint data.
    """
    goal = np.asarray(goal_points, dtype=np.float64).reshape(4, 3).copy()
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    if len(target) < 128:
        return goal, {"applied": False, "reason": "insufficient_visible_points"}
    center = np.median(target, axis=0)
    centered = target - center
    _, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
    normal = vectors[:, 0]
    camera_ray = np.asarray(camera_position, dtype=np.float64) - center
    if np.dot(normal, camera_ray) < 0:
        normal = -normal
    residual = centered @ normal
    median = float(np.median(residual))
    high = float(np.quantile(residual, 0.98))
    prominence = high - median
    if prominence < 0.012:
        return goal, {
            "applied": False, "reason": "no_12mm_visible_protrusion",
            "prominence_m": prominence,
        }
    cutoff = max(median + 0.010, float(np.quantile(residual, 0.94)))
    protruding = target[residual >= cutoff]
    if len(protruding) < 24:
        return goal, {
            "applied": False, "reason": "protrusion_too_small",
            "prominence_m": prominence, "n_points": int(len(protruding)),
        }
    # Keep the official prediction's preferred vertical location when several
    # protrusions are visible.
    delta = protruding - goal[3]
    score = np.abs(delta[:, 2]) + 0.15 * np.linalg.norm(delta[:, :2], axis=1)
    anchor = protruding[int(np.argmin(score))]
    shift = anchor - goal[3]
    goal += shift[None, :]
    return goal, {
        "applied": True,
        "selection": "target_link_dominant_plane_camera_facing_residual",
        "prominence_m": prominence,
        "cutoff_m": cutoff,
        "n_points": int(len(protruding)),
        "anchor_world": anchor.tolist(),
        "goal_translation_m": shift.tolist(),
    }


def rigid_goal_pose(current_gripper_world, current_pose_world, goal_gripper_world):
    """Fit the closest rigid gripper pose to ArticuBot's four predicted goal points."""
    source = np.asarray(current_gripper_world, dtype=np.float64).reshape(4, 3)
    target = np.asarray(goal_gripper_world, dtype=np.float64).reshape(4, 3)
    src_c = source.mean(axis=0)
    dst_c = target.mean(axis=0)
    U, _, Vt = np.linalg.svd((source - src_c).T @ (target - dst_c))
    R_delta = Vt.T @ U.T
    if np.linalg.det(R_delta) < 0:
        Vt[-1] *= -1
        R_delta = Vt.T @ U.T
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_delta @ np.asarray(current_pose_world, dtype=np.float64)[:3, :3]
    # Index 3 is panda_grasptarget in the official observation order.
    T[:3, 3] = target[3]
    fitted = (source - src_c) @ R_delta.T + dst_c
    return T, float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))


def estimate_outward_normal(point, target_points, camera_position, k=96):
    point = np.asarray(point, dtype=np.float64).reshape(3)
    points = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    distances = np.linalg.norm(points - point[None, :], axis=1)
    local = points[np.argsort(distances)[: min(int(k), len(points))]]
    centered = local - local.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered / max(len(local), 1))
    normal = vectors[:, 0]
    normal /= max(np.linalg.norm(normal), 1e-12)
    if np.dot(normal, np.asarray(camera_position) - point) < 0:
        normal = -normal
    return normal


def _pose_interpolate(T0, T1, fractions):
    T0 = np.asarray(T0, dtype=np.float64)
    T1 = np.asarray(T1, dtype=np.float64)
    key_rots = Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
    slerp = Slerp([0.0, 1.0], key_rots)
    for alpha, R in zip(fractions, slerp(fractions).as_matrix()):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
        yield T


def _gripper_points_at_pose(local_gripper_points, T):
    local = np.asarray(local_gripper_points, dtype=np.float64).reshape(-1, 3)
    return local @ T[:3, :3].T + T[:3, 3]


def _rotate_about_axis(vector, axis, angle):
    """Rodrigues rotation used to fan out purely geometric reach corridors."""
    vector = np.asarray(vector, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(np.linalg.norm(axis), 1e-12)
    return (
        vector * np.cos(float(angle))
        + np.cross(axis, vector) * np.sin(float(angle))
        + axis * np.dot(axis, vector) * (1.0 - np.cos(float(angle)))
    )


@dataclass(frozen=True)
class ReachPlan:
    goal_pose: np.ndarray
    waypoints: tuple[np.ndarray, ...]
    outward_normal: np.ndarray
    predicted_clearance_m: float
    goal_fit_rmse_m: float
    approach_offset_m: float
    candidate_index: int
    predicted_palm_clearance_m: float
    predicted_fingertip_mean_distance_m: float
    grasp_geometry_score: float


def plan_collision_aware_reach(*, current_pose, current_gripper_points, goal_points,
                               target_points, obstacle_points, camera_position,
                               approach_offset=0.09, candidate_rank=0,
                               visible_free_edge_grasp=False):
    """Choose an observed-point-cloud-clear approach corridor.

    This checks the four official gripper keypoints plus a 25 mm safety shell
    along interpolated waypoint segments.  The final 35 mm of approach is exempt
    because deliberate target contact is required.
    """
    goal_pose, fit_rmse = rigid_goal_pose(current_gripper_points, current_pose, goal_points)
    target = np.asarray(target_points, dtype=np.float64)
    nearest = target[np.argmin(np.linalg.norm(target - goal_pose[:3, 3], axis=1))]
    # Keep the model's target but prevent a floating goal farther than 25 mm from
    # the observed target-link surface.
    if np.linalg.norm(goal_pose[:3, 3] - nearest) > 0.025:
        goal_pose[:3, 3] = nearest
    surface_normal = estimate_outward_normal(
        goal_pose[:3, 3], target, camera_position
    )
    outward = surface_normal.copy()
    # The frozen camera and Panda are often on opposite sides of the object.
    # A camera-facing pregrasp can therefore be outside the arm workspace.  Keep
    # the same perceptual surface normal but orient it toward the current EEF,
    # which is the physically reachable side and uses no object metadata.
    robot_ray = np.asarray(current_pose, dtype=np.float64)[:3, 3] - goal_pose[:3, 3]
    robot_ray /= max(np.linalg.norm(robot_ray), 1e-12)
    if np.dot(outward, robot_ray) < 0:
        outward = -outward
        surface_normal = -surface_normal
    centered_target = target - target.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(
        centered_target.T @ centered_target / max(len(target), 1)
    )
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if np.dot(major_axis, np.array([0.0, 0.0, 1.0])) < 0:
        major_axis = -major_axis
    ordered_eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))
    edge_like = bool(
        ordered_eigenvalues[-1] / max(ordered_eigenvalues[-2], 1e-12) > 25.0
    )
    # Use the learned right/left fingertip correspondence for broad surfaces.
    # For an edge-like visible link, that prediction often collapses onto the
    # long edge (vertical closing, no possible two-finger pinch); the only
    # geometry-consistent closing axis is the in-plane direction orthogonal to
    # both the edge and approach.
    learned_closing = np.asarray(goal_points, dtype=np.float64)[2] - np.asarray(
        goal_points, dtype=np.float64
    )[1]
    if visible_free_edge_grasp:
        # For a visually exposed panel edge, approach laterally from outside
        # the observed link and close across its surface normal.  Approaching
        # along the panel normal makes the Panda palm hit before its fingers
        # can wrap around the edge.
        edge_direction = goal_pose[:3, 3] - target.mean(axis=0)
        edge_direction -= np.dot(edge_direction, surface_normal) * surface_normal
        if np.linalg.norm(edge_direction) > 1e-6:
            outward = edge_direction / np.linalg.norm(edge_direction)
        learned_closing = surface_normal.copy()
    z_axis = -outward
    learned_closing = learned_closing - np.dot(learned_closing, z_axis) * z_axis
    if edge_like:
        learned_closing = np.cross(major_axis, z_axis)
    if np.linalg.norm(learned_closing) < 1e-6:
        learned_closing = np.asarray(current_pose, dtype=np.float64)[:3, 1]
        learned_closing = learned_closing - np.dot(learned_closing, z_axis) * z_axis
    y_axis = learned_closing / max(np.linalg.norm(learned_closing), 1e-12)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-12)
    goal_pose[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    camera_ray = np.asarray(camera_position, dtype=np.float64) - goal_pose[:3, 3]
    camera_ray /= max(np.linalg.norm(camera_ray), 1e-12)
    predicted_axis = goal_pose[:3, 2].copy()
    if np.dot(predicted_axis, camera_ray) < 0:
        predicted_axis = -predicted_axis
    predicted_axis /= max(np.linalg.norm(predicted_axis), 1e-12)
    # Near edge-on views, the visible target is often a long, thin line.  A
    # single normal corridor can place the wrist behind the cabinet even when
    # the link itself is reachable.  Fan the EEF-facing ray around the link's
    # visible major axis; this is determined only from the target-mask cloud.
    radial = robot_ray - np.dot(robot_ray, major_axis) * major_axis
    radial /= max(np.linalg.norm(radial), 1e-12)
    if visible_free_edge_grasp:
        candidates = [
            outward,
            _rotate_about_axis(outward, surface_normal, np.deg2rad(20.0)),
            _rotate_about_axis(outward, surface_normal, np.deg2rad(-20.0)),
        ]
        approach_offset = min(float(approach_offset), 0.18)
    elif edge_like:
        candidates = [
            _rotate_about_axis(radial, major_axis, np.deg2rad(angle))
            for angle in (75.0, -75.0, 55.0, -55.0, 30.0, -30.0, 0.0)
        ]
        approach_offset = max(float(approach_offset), 0.35)
    else:
        candidates = [
            outward,
            robot_ray,
            (outward + robot_ray) / max(np.linalg.norm(outward + robot_ray), 1e-12),
            predicted_axis,
        ]
        approach_offset = min(float(approach_offset), 0.22)
    obstacle = np.asarray(obstacle_points, dtype=np.float64).reshape(-1, 3)
    current_pose = np.asarray(current_pose, dtype=np.float64)
    inv_current = np.linalg.inv(current_pose)
    current_points_h = np.concatenate(
        [np.asarray(current_gripper_points), np.ones((4, 1))], axis=1
    )
    local = (current_points_h @ inv_current.T)[:, :3]
    options = []
    fractions = np.linspace(0.0, 1.0, 13)
    for index, direction in enumerate(candidates):
        pre = goal_pose.copy()
        pre[:3, 3] = goal_pose[:3, 3] + float(approach_offset) * direction
        lift = current_pose.copy()
        lift[:3, 3] = current_pose[:3, 3] + np.array([0.0, 0.0, 0.10])
        corridor = pre.copy()
        corridor[:3, 3] = pre[:3, 3] + np.array([0.0, 0.0, 0.08])
        path = (lift, corridor, pre)
        min_clearance = np.inf
        prev = current_pose
        for waypoint in path:
            for pose in _pose_interpolate(prev, waypoint, fractions):
                gp = _gripper_points_at_pose(local, pose)
                if len(obstacle):
                    min_clearance = min(min_clearance, float(cdist(gp, obstacle).min()) - 0.025)
            prev = waypoint
        # Preserve deterministic order on effectively equal clearances.  The
        # first corridor is the empirically necessary around-edge path, not an
        # object-specific location or semantic handle prior.
        score = -float(index) if edge_like else min_clearance
        options.append((score, min_clearance, index, path))
    options.sort(key=lambda item: (-item[0], item[2]))
    rank = int(candidate_rank)
    if rank < 0 or rank >= len(options):
        raise IndexError(f"reach candidate rank {rank} outside {len(options)} candidates")
    _, clearance, index, path = options[rank]
    goal_gripper_points = _gripper_points_at_pose(local, goal_pose)
    target_tree = cKDTree(target)
    obstacle_tree = cKDTree(obstacle) if len(obstacle) else None
    palm_clearance = (
        float(obstacle_tree.query(goal_gripper_points[0], k=1)[0])
        if obstacle_tree is not None else float("inf")
    )
    fingertip_distance = float(
        np.mean(target_tree.query(goal_gripper_points[1:3], k=1)[0])
    )
    # Purely perceptual feasibility proxy: the fingertips should bracket the
    # selected link while the hand body remains clear.  It is used only to
    # order official/masked goal hypotheses, never as a success label.
    geometry_score = palm_clearance - 0.5 * fingertip_distance
    return ReachPlan(
        goal_pose=goal_pose,
        waypoints=tuple(path),
        outward_normal=np.asarray(candidates[index]),
        predicted_clearance_m=float(clearance),
        goal_fit_rmse_m=float(fit_rmse),
        approach_offset_m=float(approach_offset),
        candidate_index=int(index),
        predicted_palm_clearance_m=palm_clearance,
        predicted_fingertip_mean_distance_m=fingertip_distance,
        grasp_geometry_score=float(geometry_score),
    )


def visible_gap_metric(target_points, rest_points):
    """Robust visible separation between the selected link and the rest of the object."""
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    rest = np.asarray(rest_points, dtype=np.float64).reshape(-1, 3)
    if len(target) == 0 or len(rest) == 0:
        return None
    # Bounded deterministic subsampling keeps the feedback inexpensive.
    target = target[:: max(1, len(target) // 512)][:512]
    rest = rest[:: max(1, len(rest) // 1024)][:1024]
    nearest = cdist(target, rest).min(axis=1)
    return float(np.quantile(nearest, 0.75))
