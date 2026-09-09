
import numpy as np


BACKEND_VERSION = (
    "where2act_backend_v2_physical_20260901"
)


# ============================================================
# Panda PD
# ============================================================

ARM_STIFFNESS = 1800.0
ARM_DAMPING = 360.0

FINGER_STIFFNESS = 8000.0
FINGER_DAMPING = 1600.0


# ============================================================
# Contact
# ============================================================

CONTACT_STATIC_FRICTION = 2.0
CONTACT_DYNAMIC_FRICTION = 2.0
CONTACT_RESTITUTION = 0.0


# ============================================================
# Object target joint
# ============================================================

TARGET_LOCK_STIFFNESS = 5000.0
TARGET_LOCK_DAMPING = 500.0

TARGET_FREE_STIFFNESS = 0.0
TARGET_FREE_DAMPING = 0.05


def set_target_free(
    obj,
    target_joint,
    target_index,
):
    """
    OPEN / TO_GRASP / HOLD / OPERATE：

        target K = 0
        target D = 0.05

    与学长新底层的 operate-like target drive 对齐。
    """

    index = int(
        target_index
    )

    qpos = np.asarray(
        obj.get_qpos(),
        dtype=np.float64,
    )

    q_now = float(
        qpos[
            index
        ]
    )

    target_joint.set_drive_property(
        stiffness=float(
            TARGET_FREE_STIFFNESS
        ),
        damping=float(
            TARGET_FREE_DAMPING
        ),
    )

    # K=0 时这个位置 target 不产生恢复弹簧力，
    # 但仍同步为当前值，避免旧 drive target 残留。
    target_joint.set_drive_target(
        q_now
    )

    target_joint.set_drive_velocity_target(
        0.0
    )

    return q_now


def lock_target_to_q(
    obj,
    target_joint,
    target_index,
    target_q,
):
    """
    INIT/PREGRASP/CLOSE：

    target 回到 benchmark 规定的初始 q，
    qvel 清零，
    再使用 K=5000,D=500 保持。
    """

    index = int(
        target_index
    )

    target_q = float(
        target_q
    )

    qpos = np.asarray(
        obj.get_qpos(),
        dtype=np.float64,
    ).copy()

    qpos[
        index
    ] = target_q

    obj.set_qpos(
        qpos
    )

    try:

        qvel = np.asarray(
            obj.get_qvel(),
            dtype=np.float64,
        ).copy()

        if index < len(
            qvel
        ):

            qvel[
                index
            ] = 0.0

            obj.set_qvel(
                qvel
            )

    except Exception:
        pass


    target_joint.set_drive_property(
        stiffness=float(
            TARGET_LOCK_STIFFNESS
        ),
        damping=float(
            TARGET_LOCK_DAMPING
        ),
    )

    target_joint.set_drive_target(
        target_q
    )

    target_joint.set_drive_velocity_target(
        0.0
    )


def apply_target_contact_material(
    target_link,
    physical_material,
):
    """
    将 target link collision shapes 设置为与 Panda fingers
    相同的物理材料。

    这是普通 friction/contact material，
    不是 weld、drive 或 suction。
    """

    shapes = list(
        target_link.get_collision_shapes()
    )

    for shape in shapes:

        shape.set_physical_material(
            physical_material
        )

    return {
        "num_target_collision_shapes":
            int(
                len(
                    shapes
                )
            ),

        "static_friction":
            CONTACT_STATIC_FRICTION,

        "dynamic_friction":
            CONTACT_DYNAMIC_FRICTION,

        "restitution":
            CONTACT_RESTITUTION,
    }

