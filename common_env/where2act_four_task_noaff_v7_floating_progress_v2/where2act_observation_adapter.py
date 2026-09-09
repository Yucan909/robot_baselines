import numpy as np


class Where2ActObservationError(RuntimeError):
    pass


def capture_where2act_observation(
    scene,
    camera,
    obj,
    target_link,
    *,
    object_origin_world=None,
):
    """
    从统一 SAPIEN benchmark 的同一个固定相机中，
    构造 Where2Act 所需的单视角输入。

    返回：
        rgb                  H x W x 3
        points_camera        N x 3，SAPIEN 相机坐标
        points_world         N x 3，世界坐标
        points_model         N x 3，Where2Act 输入坐标
        candidate_mask       N，selected-link mask
        camera_to_world_R    3 x 3
        camera_position_world
        diagnostics

    注意：
        这里不做1200点采样。
        Where2ActPolicy 自己负责官方风格的
        30000 pool -> FPS 10000。
    """

    if object_origin_world is None:
        # 我们统一 benchmark 中 PartNet 根节点固定在世界原点。
        object_origin_world = np.zeros(
            3,
            dtype=np.float64,
        )

    object_origin_world = np.asarray(
        object_origin_world,
        dtype=np.float64,
    ).reshape(3)

    # ========================================================
    # 1. 使用和 FlowBot 完全相同的一帧 SAPIEN observation
    # ========================================================

    scene.update_render()
    camera.take_picture()

    rgba = np.asarray(
        camera.get_float_texture(
            "Color"
        )
    )

    rgb = np.clip(
        rgba[..., :3],
        0.0,
        1.0,
    ).astype(
        np.float32
    )

    position = np.asarray(
        camera.get_float_texture(
            "Position"
        )
    )

    segmentation = np.asarray(
        camera.get_uint32_texture(
            "Segmentation"
        )
    )

    if (
        position.ndim != 3
        or position.shape[-1] < 4
    ):
        raise Where2ActObservationError(
            f"Position texture shape异常: "
            f"{position.shape}"
        )

    if (
        segmentation.ndim != 3
        or segmentation.shape[-1] < 2
    ):
        raise Where2ActObservationError(
            f"Segmentation texture shape异常: "
            f"{segmentation.shape}"
        )

    # ========================================================
    # 2. actor segmentation
    #
    # 与当前统一 FlowBot executor 完全相同：
    # segmentation[..., 1]
    # ========================================================

    actor_seg = (
        segmentation[..., 1]
        .astype(np.int64)
    )

    object_link_ids = np.asarray(
        [
            int(link.get_id())
            for link in obj.get_links()
        ],
        dtype=np.int64,
    )

    target_actor_id = int(
        target_link.get_id()
    )

    valid = (
        position[..., 3]
        > 0
    )

    object_mask_image = np.isin(
        actor_seg,
        object_link_ids,
    )

    keep = (
        valid
        & object_mask_image
    )

    visible_object_pixels = int(
        keep.sum()
    )

    if visible_object_pixels == 0:
        raise Where2ActObservationError(
            "object_not_visible"
        )

    # ========================================================
    # 3. 所有可见物体点
    #
    # 不在这里下采样！
    # ========================================================

    points_camera = (
        position[..., :3][keep]
        .astype(np.float64)
    )

    actor_ids = (
        actor_seg[keep]
        .astype(np.int64)
    )

    candidate_mask = (
        actor_ids
        == target_actor_id
    )

    visible_target_points = int(
        candidate_mask.sum()
    )

    if visible_target_points == 0:
        raise Where2ActObservationError(
            "target_not_visible"
        )

    # ========================================================
    # 4. camera -> world
    #
    # 完全复用当前 FlowBot 已经验证过的计算：
    #
    # p_world = p_camera @ R.T + t
    # ========================================================

    model_matrix = np.asarray(
        camera.get_model_matrix(),
        dtype=np.float64,
    )

    if model_matrix.shape != (4, 4):
        raise Where2ActObservationError(
            f"camera model matrix异常: "
            f"{model_matrix.shape}"
        )

    R_camera_to_world = (
        model_matrix[:3, :3]
    )

    camera_position_world = (
        model_matrix[:3, 3]
    )

    points_world = (
        points_camera
        @ R_camera_to_world.T
        + camera_position_world
    )

    # ========================================================
    # 5. 构造 Where2Act 的模型坐标
    #
    # 原始 Where2Act：
    # camera离物体约5m，
    # 然后：
    #
    #     pc[:, 0] -= 5
    #
    # 本质是在相机坐标系中把“物体原点”移到0附近。
    #
    # 对任意固定相机更一般地写成：
    #
    # origin_camera =
    #     (origin_world - t) @ R
    #
    # points_model =
    #     points_camera - origin_camera
    #
    # 等价于：
    #
    # points_model =
    #     (points_world - origin_world) @ R
    #
    # 因而：
    # - 保留相机方向坐标系；
    # - 去掉相机与物体的绝对距离；
    # - 不改变物体真实尺度。
    # ========================================================

    object_origin_camera = (
        (
            object_origin_world
            - camera_position_world
        )
        @ R_camera_to_world
    )

    points_model = (
        points_camera
        - object_origin_camera[
            None,
            :
        ]
    )

    # 独立用world坐标再算一次，检查坐标变换。
    points_model_check = (
        (
            points_world
            - object_origin_world[
                None,
                :
            ]
        )
        @ R_camera_to_world
    )

    coordinate_consistency_error = float(
        np.max(
            np.abs(
                points_model
                - points_model_check
            )
        )
    )

    if (
        not np.isfinite(
            coordinate_consistency_error
        )
        or coordinate_consistency_error
        > 1e-5
    ):
        raise Where2ActObservationError(
            "camera/world坐标转换不一致: "
            f"{coordinate_consistency_error}"
        )

    # ========================================================
    # 6. 旋转矩阵 sanity check
    # ========================================================

    ortho_error = float(
        np.linalg.norm(
            R_camera_to_world.T
            @ R_camera_to_world
            - np.eye(3)
        )
    )

    determinant = float(
        np.linalg.det(
            R_camera_to_world
        )
    )

    if (
        ortho_error > 1e-4
        or abs(
            determinant - 1.0
        ) > 1e-4
    ):
        raise Where2ActObservationError(
            "camera rotation非法: "
            f"orth={ortho_error}, "
            f"det={determinant}"
        )

    # ========================================================
    # 7. 点云尺度检查
    # ========================================================

    point_min = (
        points_model.min(
            axis=0
        )
    )

    point_max = (
        points_model.max(
            axis=0
        )
    )

    point_extent = (
        point_max
        - point_min
    )

    if not np.all(
        np.isfinite(
            point_extent
        )
    ):
        raise Where2ActObservationError(
            "point cloud extent含NaN/Inf"
        )

    # ========================================================
    # 输出
    # ========================================================

    return {
        "rgb":
            rgb,

        "points_camera":
            points_camera.astype(
                np.float32
            ),

        "points_world":
            points_world.astype(
                np.float32
            ),

        "points_model":
            points_model.astype(
                np.float32
            ),

        "candidate_mask":
            candidate_mask.astype(
                bool
            ),

        "camera_to_world_R":
            R_camera_to_world.copy(),

        "camera_position_world":
            camera_position_world.copy(),

        "object_origin_world":
            object_origin_world.copy(),

        "object_origin_camera":
            object_origin_camera.copy(),

        "diagnostics": {
            "visible_object_points":
                visible_object_pixels,

            "visible_target_points":
                visible_target_points,

            "target_fraction":
                float(
                    visible_target_points
                    / visible_object_pixels
                ),

            "point_min_model":
                point_min.copy(),

            "point_max_model":
                point_max.copy(),

            "point_extent_model":
                point_extent.copy(),

            "camera_rotation_orthogonality_error":
                ortho_error,

            "camera_rotation_determinant":
                determinant,

            "coordinate_consistency_error":
                coordinate_consistency_error,
        },
    }
