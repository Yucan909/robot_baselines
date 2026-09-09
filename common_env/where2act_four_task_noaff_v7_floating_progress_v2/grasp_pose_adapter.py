import numpy as np


class GraspPoseError(RuntimeError):
    pass


def _normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= eps:
        raise GraspPoseError("向量无法归一化")
    return v / n


def estimate_local_surface_normal(
    contact_point,
    target_points_world,
    camera_position_world,
    *,
    k_neighbors=64,
    min_neighbors=12,
):
    """
    从当前单视角 target-link 点云估计 FlowBot contact point 附近的局部法向。

    仅使用当前观测点云，不使用 GT joint axis / 完整 CAD / 关节参数。
    PCA 最小特征值对应局部表面法向；法向符号利用相机位置消歧，
    令 normal 指向相机/可见侧。
    """
    p = np.asarray(contact_point, dtype=np.float64).reshape(3)
    pts = np.asarray(target_points_world, dtype=np.float64).reshape(-1, 3)
    cam = np.asarray(camera_position_world, dtype=np.float64).reshape(3)

    pts = pts[np.all(np.isfinite(pts), axis=1)]

    if len(pts) < int(min_neighbors):
        raise GraspPoseError(
            f"目标局部点不足: {len(pts)} < {int(min_neighbors)}"
        )

    d = np.linalg.norm(pts - p[None, :], axis=1)
    order = np.argsort(d)
    k = min(int(k_neighbors), len(pts))
    local = pts[order[:k]]

    if len(local) < int(min_neighbors):
        raise GraspPoseError("局部点数不足")

    center = local.mean(axis=0)
    X = local - center[None, :]
    C = (X.T @ X) / max(len(local), 1)

    if not np.all(np.isfinite(C)):
        raise GraspPoseError("局部协方差存在 NaN/Inf")

    values, vectors = np.linalg.eigh(C)
    normal = _normalize(vectors[:, int(np.argmin(values))])

    # normal 指向可见/相机侧。
    camera_ray = cam - p
    if float(np.dot(normal, camera_ray)) < 0.0:
        normal = -normal

    return {
        "normal": normal,
        "local_center": center,
        "eigenvalues": values,
        "neighbor_count": int(len(local)),
        "neighbor_radius_m": float(d[order[k - 1]]),
    }


def build_twofinger_grasp_pose(
    contact_point,
    pull_direction,
    target_points_world,
    camera_position_world,
    *,
    current_grasp_rotation=None,
):
    """
    将 FlowBot 的 contact point + pull direction 转为统一 parallel-jaw 6D pose。

    Panda grasp frame 约定：
      local Y = 两指开合方向；
      local Z = 工具前向/接近方向。

    固定通用规则：
      1) 用可见 target 点云 PCA 估计表面外法向 n；
      2) 工具 +Z 指向物体内部：z = -n，因此夹爪从可见侧正面接近；
      3) 将 Flow 方向投影到局部切平面，作为 x 轴（主要操作方向）；
      4) y = z × x，作为 finger closing axis。

    这样 pull 方向主要落在夹爪切向 x 轴上，真实拉动需要由 finger
    法向夹持力 + 摩擦来传递；不会使用任何人工 object constraint。

    所有类别/shape/link 使用同一规则，且不使用 GT joint axis。
    """
    p = np.asarray(contact_point, dtype=np.float64).reshape(3)
    flow = _normalize(pull_direction)

    normal_info = estimate_local_surface_normal(
        p,
        target_points_world,
        camera_position_world,
    )
    outward_normal = normal_info["normal"]
    z_axis = _normalize(-outward_normal)  # 从相机侧朝向物体

    tangent = flow - float(np.dot(flow, outward_normal)) * outward_normal
    fallback_used = False

    if not np.isfinite(np.linalg.norm(tangent)) or np.linalg.norm(tangent) <= 1e-5:
        fallback_used = True
        candidates = []

        if current_grasp_rotation is not None:
            R0 = np.asarray(current_grasp_rotation, dtype=np.float64).reshape(3, 3)
            candidates.extend([R0[:, 0], R0[:, 1]])

        candidates.extend(
            [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ]
        )

        tangent = None
        for c in candidates:
            c = np.asarray(c, dtype=np.float64).reshape(3)
            c = c - float(np.dot(c, outward_normal)) * outward_normal
            if np.linalg.norm(c) > 1e-5:
                tangent = c
                break

        if tangent is None:
            raise GraspPoseError("无法构造稳定的局部切向")

    x_axis = _normalize(tangent)
    if float(np.dot(x_axis, flow)) < 0.0:
        x_axis = -x_axis

    y_axis = _normalize(np.cross(z_axis, x_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))

    R = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(R) < 0.0:
        y_axis = -y_axis
        R = np.column_stack([x_axis, y_axis, z_axis])

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p

    return {
        "T_world_grasp": T,
        "surface_outward_normal_world": outward_normal,
        "closing_axis_world": y_axis,
        "approach_axis_world": z_axis,
        "pull_tangent_axis_world": x_axis,
        "normal_info": normal_info,
        "flow_tangent_fallback_used": bool(fallback_used),
    }
