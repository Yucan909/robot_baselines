import sys
from pathlib import Path

import numpy as np
import torch

from pointnet2_ops.pointnet2_utils import (
    furthest_point_sample,
)


HOME = Path.home()

WHERE2ACT_CODE = (
    HOME
    / "robot_baselines/repos/where2act/code"
)

if str(WHERE2ACT_CODE) not in sys.path:
    sys.path.insert(
        0,
        str(WHERE2ACT_CODE),
    )


from models.model_3d import Network


# ============================================================
# 固定的官方 Where2Act 3D 参数
# ============================================================

W2A_FEAT_DIM = 128
W2A_RV_DIM = 10
W2A_RV_CNT = 100

# 官方训练 / 推理点数
W2A_NUM_POINTS = 10000

# 官方可视化代码会先组织约 30000 点，再 FPS 到 10000。
W2A_PRE_FPS_POOL = 30000

# 官方 gripper pose：
#
# gripper position =
#     interaction point - gripper_up * 0.1
#
# 即夹爪基座位于接触点后方 10 cm。
W2A_GRIPPER_BACKOFF = 0.10


class Where2ActPolicyError(RuntimeError):
    pass


class Where2ActPolicy:
    """
    Where2Act 3D policy 的统一推理封装。

    输入：
        points_model:
            N x 3
            Where2Act 网络坐标系下点云。

        points_world:
            N x 3
            与 points_model 一一对应的世界坐标。

        camera_to_world_R:
            3 x 3
            相机方向向量 -> 世界方向向量。

        candidate_mask:
            N bool，可选。
            unified benchmark 中可用 target-link identity
            限制允许交互的目标 link。

    输出：
        interaction point
        actionability score
        gripper orientation
        critic score
        4x4 world gripper target pose

    注意：
        本类不负责机器人运动。
        motion planning 在下一层完成。
    """

    def __init__(
        self,
        checkpoint=None,
        *,
        device="cuda:0",
        feat_dim=W2A_FEAT_DIM,
        rv_dim=W2A_RV_DIM,
        rv_cnt=W2A_RV_CNT,
        allow_untrained=False,
    ):
        self.device = torch.device(
            device
        )

        if (
            self.device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise Where2ActPolicyError(
                "CUDA requested but unavailable"
            )

        self.feat_dim = int(
            feat_dim
        )

        self.rv_dim = int(
            rv_dim
        )

        self.rv_cnt = int(
            rv_cnt
        )

        self.allow_untrained = bool(
            allow_untrained
        )

        self.network = Network(
            self.feat_dim,
            self.rv_dim,
            self.rv_cnt,
        )

        self.network.to(
            self.device
        )

        self.network.eval()

        self.checkpoint = None
        self.is_trained = False

        if checkpoint is not None:
            self.load_checkpoint(
                checkpoint
            )

        elif not self.allow_untrained:
            raise Where2ActPolicyError(
                "未提供 checkpoint。"
                "正式推理禁止使用随机权重。"
            )


    # ========================================================
    # checkpoint
    # ========================================================

    def load_checkpoint(
        self,
        checkpoint,
    ):
        checkpoint = (
            Path(checkpoint)
            .expanduser()
            .resolve()
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                checkpoint
            )

        data = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        # Where2Act官方通常直接保存network.state_dict()
        if (
            isinstance(data, dict)
            and "state_dict" in data
            and isinstance(
                data["state_dict"],
                dict,
            )
        ):
            state = data[
                "state_dict"
            ]
        else:
            state = data

        # 兼容可能带 module. 前缀的情况
        clean_state = {}

        for key, value in state.items():

            new_key = str(
                key
            )

            if new_key.startswith(
                "module."
            ):
                new_key = new_key[
                    len("module.") :
                ]

            clean_state[
                new_key
            ] = value

        result = (
            self.network
            .load_state_dict(
                clean_state,
                strict=True,
            )
        )

        self.network.to(
            self.device
        )

        self.network.eval()

        self.checkpoint = str(
            checkpoint
        )

        self.is_trained = True

        print(
            "Where2Act checkpoint loaded:",
            checkpoint,
        )

        print(
            "missing:",
            result.missing_keys,
        )

        print(
            "unexpected:",
            result.unexpected_keys,
        )


    # ========================================================
    # 输入检查
    # ========================================================

    @staticmethod
    def _validate_points(
        points,
        name,
    ):
        points = np.asarray(
            points,
            dtype=np.float32,
        )

        if (
            points.ndim != 2
            or points.shape[1] != 3
        ):
            raise Where2ActPolicyError(
                f"{name} 应为 N x 3，"
                f"实际 {points.shape}"
            )

        if len(points) < 2:
            raise Where2ActPolicyError(
                f"{name} 点数过少"
            )

        if not np.all(
            np.isfinite(
                points
            )
        ):
            raise Where2ActPolicyError(
                f"{name} 包含 NaN/Inf"
            )

        return points


    # ========================================================
    # 官方式 30000 -> FPS 10000
    # ========================================================

    def _sample_points(
        self,
        points_model,
        points_world,
        candidate_mask,
        *,
        seed,
    ):
        n = int(
            len(
                points_model
            )
        )

        rng = np.random.default_rng(
            int(seed)
        )

        all_idx = np.arange(
            n,
            dtype=np.int64,
        )

        # ----------------------------------------------------
        # Where2Act原代码：
        # shuffle之后反复扩增idx直到>=30000，
        # 然后FPS到10000。
        # ----------------------------------------------------

        shuffled = (
            rng.permutation(
                all_idx
            )
        )

        if n >= W2A_PRE_FPS_POOL:

            pool_idx = shuffled[
                :W2A_PRE_FPS_POOL
            ]

        else:

            repeat_count = int(
                np.ceil(
                    W2A_PRE_FPS_POOL
                    / n
                )
            )

            pool_idx = np.tile(
                shuffled,
                repeat_count,
            )[
                :W2A_PRE_FPS_POOL
            ]

        pool_model = (
            points_model[
                pool_idx
            ]
        )

        pool_tensor = (
            torch
            .from_numpy(
                pool_model
            )
            .to(
                self.device
            )
            .unsqueeze(
                0
            )
            .contiguous()
        )

        with torch.no_grad():

            fps_idx = (
                furthest_point_sample(
                    pool_tensor,
                    W2A_NUM_POINTS,
                )
                .long()
                .reshape(
                    -1
                )
            )

        fps_idx_np = (
            fps_idx
            .detach()
            .cpu()
            .numpy()
        )

        sampled_original_idx = (
            pool_idx[
                fps_idx_np
            ]
        )

        sampled_model = (
            points_model[
                sampled_original_idx
            ]
        )

        sampled_world = (
            points_world[
                sampled_original_idx
            ]
        )

        if candidate_mask is None:

            sampled_mask = np.ones(
                W2A_NUM_POINTS,
                dtype=bool,
            )

        else:

            sampled_mask = (
                candidate_mask[
                    sampled_original_idx
                ]
            )

            if not np.any(
                sampled_mask
            ):
                raise Where2ActPolicyError(
                    "target_not_sampled"
                )

        return {
            "points_model":
                sampled_model,

            "points_world":
                sampled_world,

            "candidate_mask":
                sampled_mask,

            "original_indices":
                sampled_original_idx,
        }


    # ========================================================
    # 把query point放到第0位
    #
    # Where2Act actor / critic 的约定就是：
    # 第0个点 = 当前交互query。
    # ========================================================

    @staticmethod
    def _move_query_to_front(
        points,
        query_idx,
    ):
        points = points.clone()

        query_idx = int(
            query_idx
        )

        if query_idx == 0:
            return points

        first = (
            points[
                :,
                0,
                :
            ]
            .clone()
        )

        query = (
            points[
                :,
                query_idx,
                :
            ]
            .clone()
        )

        points[
            :,
            0,
            :
        ] = query

        points[
            :,
            query_idx,
            :
        ] = first

        return points


    # ========================================================
    # 预测
    # ========================================================

    def predict(
        self,
        points_model,
        points_world,
        camera_to_world_R,
        *,
        candidate_mask=None,
        seed=0,
    ):
        if (
            not self.is_trained
            and not self.allow_untrained
        ):
            raise Where2ActPolicyError(
                "禁止使用未训练模型进行正式推理"
            )

        points_model = (
            self._validate_points(
                points_model,
                "points_model",
            )
        )

        points_world = (
            self._validate_points(
                points_world,
                "points_world",
            )
        )

        if (
            points_model.shape
            != points_world.shape
        ):
            raise Where2ActPolicyError(
                "points_model 与 points_world "
                "必须一一对应"
            )

        camera_to_world_R = np.asarray(
            camera_to_world_R,
            dtype=np.float64,
        )

        if (
            camera_to_world_R.shape
            != (3, 3)
        ):
            raise Where2ActPolicyError(
                "camera_to_world_R 应为3x3"
            )

        if candidate_mask is not None:

            candidate_mask = np.asarray(
                candidate_mask,
                dtype=bool,
            ).reshape(
                -1
            )

            if len(
                candidate_mask
            ) != len(
                points_model
            ):
                raise Where2ActPolicyError(
                    "candidate_mask长度错误"
                )

            if not np.any(
                candidate_mask
            ):
                raise Where2ActPolicyError(
                    "target_not_visible"
                )


        # ----------------------------------------------------
        # 1. 30000 pool -> FPS 10000
        # ----------------------------------------------------

        sampled = (
            self._sample_points(
                points_model,
                points_world,
                candidate_mask,
                seed=seed,
            )
        )

        sampled_model = sampled[
            "points_model"
        ]

        sampled_world = sampled[
            "points_world"
        ]

        sampled_mask = sampled[
            "candidate_mask"
        ]


        pc = (
            torch
            .from_numpy(
                sampled_model
            )
            .to(
                self.device
            )
            .unsqueeze(
                0
            )
            .contiguous()
        )


        # ----------------------------------------------------
        # 2. Actionability
        # ----------------------------------------------------

        with torch.inference_mode():

            action_scores = (
                self.network
                .inference_action_score(
                    pc
                )[
                    0
                ]
            )


        if not torch.isfinite(
            action_scores
        ).all():

            raise Where2ActPolicyError(
                "action score包含NaN/Inf"
            )


        mask_tensor = (
            torch
            .from_numpy(
                sampled_mask
            )
            .to(
                self.device
            )
        )


        masked_scores = (
            action_scores.clone()
        )

        masked_scores[
            ~mask_tensor
        ] = -torch.inf


        query_idx = int(
            torch.argmax(
                masked_scores
            ).item()
        )


        interaction_score = float(
            action_scores[
                query_idx
            ].item()
        )


        interaction_point_model = (
            sampled_model[
                query_idx
            ].astype(
                np.float64
            )
        )


        interaction_point_world = (
            sampled_world[
                query_idx
            ].astype(
                np.float64
            )
        )


        # ----------------------------------------------------
        # 3. Query point移到0号
        # ----------------------------------------------------

        query_pc = (
            self._move_query_to_front(
                pc,
                query_idx,
            )
        )


        # ----------------------------------------------------
        # 4. Actor：
        # 生成100个orientation proposal
        # ----------------------------------------------------

        with torch.inference_mode():

            pred_6d = (
                self.network
                .inference_actor(
                    query_pc
                )[
                    0
                ]
            )


            pred_R = (
                self.network.actor.bgs(
                    pred_6d.reshape(
                        -1,
                        3,
                        2,
                    )
                )
            )


        if (
            pred_R.shape
            != (
                self.rv_cnt,
                3,
                3,
            )
        ):
            raise Where2ActPolicyError(
                f"Actor rotation shape异常: "
                f"{pred_R.shape}"
            )


        # ----------------------------------------------------
        # 5. Critic：
        #
        # 官方 inference_critic 本质上：
        #
        # PointNet(query_pc)
        # → 第0点feature
        # → Critic(feature, dirs1, dirs2)
        #
        # 这里一次计算100个candidate，
        # 避免重复跑100遍PointNet++。
        #
        # 数学上与官方critic完全相同。
        # ----------------------------------------------------

        with torch.inference_mode():

            whole_feats = (
                self.network
                .pointnet2(
                    query_pc.repeat(
                        1,
                        1,
                        2,
                    )
                )
            )


            query_feat = (
                whole_feats[
                    :,
                    :,
                    0
                ]
            )


            expanded_feat = (
                query_feat
                .expand(
                    self.rv_cnt,
                    -1,
                )
                .contiguous()
            )


            dirs1 = (
                pred_R[
                    :,
                    :,
                    0
                ]
            )


            dirs2 = (
                pred_R[
                    :,
                    :,
                    1
                ]
            )


            query_dirs = torch.cat(
                [
                    dirs1,
                    dirs2,
                ],
                dim=1,
            )


            critic_logits = (
                self.network
                .critic(
                    expanded_feat,
                    query_dirs,
                )
            )


            critic_scores = (
                torch.sigmoid(
                    critic_logits
                )
            )


        if not torch.isfinite(
            critic_scores
        ).all():

            raise Where2ActPolicyError(
                "critic score包含NaN/Inf"
            )


        best_proposal_idx = int(
            torch.argmax(
                critic_scores
            ).item()
        )


        best_critic_score = float(
            critic_scores[
                best_proposal_idx
            ].item()
        )


        best_R_camera = (
            pred_R[
                best_proposal_idx
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )


        # ====================================================
        # 6. 官方 Where2Act pose conversion
        #
        # pred_R的：
        #
        # column 0 = gripper_direction / up
        # column 1 = forward
        #
        # 官方代码：
        #
        # up      = R_cam_world @ up
        # forward = R_cam_world @ forward
        #
        # left = cross(up, forward)
        # forward = cross(left, up)
        #
        # rot columns:
        # [forward, left, up]
        #
        # position:
        # interaction - up * 0.1
        # ====================================================

        up_camera = (
            best_R_camera[
                :,
                0
            ]
        )


        forward_camera = (
            best_R_camera[
                :,
                1
            ]
        )


        up_world = (
            camera_to_world_R
            @ up_camera
        )


        forward_world_raw = (
            camera_to_world_R
            @ forward_camera
        )


        up_world = (
            up_world
            / (
                np.linalg.norm(
                    up_world
                )
                + 1e-12
            )
        )


        forward_world_raw = (
            forward_world_raw
            / (
                np.linalg.norm(
                    forward_world_raw
                )
                + 1e-12
            )
        )


        left_world = np.cross(
            up_world,
            forward_world_raw,
        )


        left_norm = float(
            np.linalg.norm(
                left_world
            )
        )


        if left_norm < 1e-8:
            raise Where2ActPolicyError(
                "Actor给出的两个方向近乎平行"
            )


        left_world = (
            left_world
            / left_norm
        )


        forward_world = np.cross(
            left_world,
            up_world,
        )


        forward_world = (
            forward_world
            / (
                np.linalg.norm(
                    forward_world
                )
                + 1e-12
            )
        )


        grasp_R_world = np.column_stack(
            [
                forward_world,
                left_world,
                up_world,
            ]
        )


        grasp_position_world = (
            interaction_point_world
            - up_world
            * W2A_GRIPPER_BACKOFF
        )


        grasp_pose_world = np.eye(
            4,
            dtype=np.float64,
        )


        grasp_pose_world[
            :3,
            :3
        ] = grasp_R_world


        grasp_pose_world[
            :3,
            3
        ] = grasp_position_world


        # ----------------------------------------------------
        # Rotation sanity
        # ----------------------------------------------------

        ortho_error = float(
            np.linalg.norm(
                grasp_R_world.T
                @ grasp_R_world
                - np.eye(
                    3
                )
            )
        )


        determinant = float(
            np.linalg.det(
                grasp_R_world
            )
        )


        if (
            ortho_error > 1e-4
            or determinant < 0.99
            or determinant > 1.01
        ):
            raise Where2ActPolicyError(
                "输出旋转矩阵不合法: "
                f"orth={ortho_error}, "
                f"det={determinant}"
            )


        return {
            "interaction_point_model":
                interaction_point_model,

            "interaction_point_world":
                interaction_point_world,

            "interaction_score":
                interaction_score,

            "sampled_query_index":
                query_idx,

            "original_query_index":
                int(
                    sampled[
                        "original_indices"
                    ][
                        query_idx
                    ]
                ),

            "proposal_index":
                best_proposal_idx,

            "critic_score":
                best_critic_score,

            "up_camera":
                up_camera,

            "forward_camera":
                forward_camera,

            "up_world":
                up_world,

            "forward_world":
                forward_world,

            "left_world":
                left_world,

            "grasp_pose_world":
                grasp_pose_world,

            "grasp_position_world":
                grasp_position_world,

            "rotation_orthogonality_error":
                ortho_error,

            "rotation_determinant":
                determinant,

            "num_input_points":
                int(
                    len(
                        points_model
                    )
                ),

            "num_model_points":
                W2A_NUM_POINTS,

            "checkpoint":
                self.checkpoint,

            "trained":
                self.is_trained,
        }
