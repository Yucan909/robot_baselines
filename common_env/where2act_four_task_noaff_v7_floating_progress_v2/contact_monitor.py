import numpy as np


class ContactMonitorError(RuntimeError):
    pass


def _actor_id(actor):
    """兼容 SAPIEN 1.1 的 get_id()/id 两种接口。"""
    if hasattr(actor, "get_id"):
        return int(actor.get_id())
    return int(actor.id)


def _impulse_norm(point):
    impulse = np.asarray(point.impulse, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(impulse)):
        return 0.0
    return float(np.linalg.norm(impulse))


def _contact_impulse_between(scene, link_a, link_b, *, impulse_epsilon=1e-8):
    """
    返回当前 physics step 中 link_a 与 link_b 的有效接触信息。

    SAPIEN 的 Contact 在接触即将开始/结束时也可能存在，
    所以不能仅看 Contact 对象存在；至少要求 ContactPoint impulse 非零。
    """
    aid = _actor_id(link_a)
    bid = _actor_id(link_b)

    total_impulse = 0.0
    max_impulse = 0.0
    effective_points = 0
    positions = []

    for contact in scene.get_contacts():
        a0 = _actor_id(contact.actor0)
        a1 = _actor_id(contact.actor1)

        if not (
            (a0 == aid and a1 == bid)
            or (a0 == bid and a1 == aid)
        ):
            continue

        for point in contact.points:
            impulse = _impulse_norm(point)
            if impulse <= float(impulse_epsilon):
                continue

            effective_points += 1
            total_impulse += impulse
            max_impulse = max(max_impulse, impulse)
            try:
                positions.append(
                    np.asarray(point.position, dtype=np.float64).reshape(3)
                )
            except Exception:
                pass

    return {
        "effective": bool(effective_points > 0),
        "effective_points": int(effective_points),
        "total_impulse": float(total_impulse),
        "max_impulse": float(max_impulse),
        "positions": positions,
    }


def read_twofinger_contact(
    scene,
    left_finger_link,
    right_finger_link,
    target_link,
    *,
    impulse_epsilon=1e-8,
):
    left = _contact_impulse_between(
        scene,
        left_finger_link,
        target_link,
        impulse_epsilon=impulse_epsilon,
    )
    right = _contact_impulse_between(
        scene,
        right_finger_link,
        target_link,
        impulse_epsilon=impulse_epsilon,
    )

    return {
        "left_contact": bool(left["effective"]),
        "right_contact": bool(right["effective"]),
        "bilateral_contact": bool(
            left["effective"] and right["effective"]
        ),
        "any_finger_contact": bool(
            left["effective"] or right["effective"]
        ),
        "left_total_impulse": float(left["total_impulse"]),
        "right_total_impulse": float(right["total_impulse"]),
        "left_max_impulse": float(left["max_impulse"]),
        "right_max_impulse": float(right["max_impulse"]),
        "left_contact_points": int(left["effective_points"]),
        "right_contact_points": int(right["effective_points"]),
    }


def monitor_grasp_establishment(
    controller,
    target_link,
    *,
    settle_steps=300,
    tail_steps=100,
    min_bilateral_fraction=0.50,
    impulse_epsilon=1e-8,
):
    """
    闭合后的真实抓取稳定性判定。

    固定规则：
    - 保持两指闭合命令；
    - 总共仿真 settle_steps；
    - 只统计最后 tail_steps，避免刚闭合时的瞬态碰撞；
    - tail window 中 >= min_bilateral_fraction 的 physics frames
      同时存在左右手指 -> target link 的非零冲量接触，才算 firm grasp。

    该指标与最终开门结果完全解耦。
    """
    settle_steps = int(settle_steps)
    tail_steps = int(tail_steps)

    if settle_steps <= 0:
        raise ValueError("settle_steps 必须 > 0")
    if tail_steps <= 0 or tail_steps > settle_steps:
        raise ValueError("tail_steps 必须位于 (0, settle_steps]")

    frames = []

    for _ in range(settle_steps):
        controller.keep_gripper_closed()
        controller.step()

        frames.append(
            read_twofinger_contact(
                controller.scene,
                controller.left_finger_link,
                controller.right_finger_link,
                target_link,
                impulse_epsilon=impulse_epsilon,
            )
        )

    tail = frames[-tail_steps:]

    bilateral_count = sum(
        int(x["bilateral_contact"])
        for x in tail
    )
    any_count = sum(
        int(x["any_finger_contact"])
        for x in tail
    )
    left_count = sum(
        int(x["left_contact"])
        for x in tail
    )
    right_count = sum(
        int(x["right_contact"])
        for x in tail
    )

    bilateral_fraction = bilateral_count / float(tail_steps)
    any_fraction = any_count / float(tail_steps)

    max_left_impulse = max(
        (x["left_max_impulse"] for x in tail),
        default=0.0,
    )
    max_right_impulse = max(
        (x["right_max_impulse"] for x in tail),
        default=0.0,
    )

    firm_grasp = bool(
        bilateral_fraction >= float(min_bilateral_fraction)
    )

    return {
        "firm_grasp": firm_grasp,
        "settle_steps": settle_steps,
        "tail_steps": tail_steps,
        "required_bilateral_fraction": float(min_bilateral_fraction),
        "bilateral_frames": int(bilateral_count),
        "bilateral_fraction": float(bilateral_fraction),
        "any_contact_fraction": float(any_fraction),
        "left_contact_fraction": float(left_count / float(tail_steps)),
        "right_contact_fraction": float(right_count / float(tail_steps)),
        "max_left_impulse": float(max_left_impulse),
        "max_right_impulse": float(max_right_impulse),
        "final_frame": frames[-1] if frames else None,
    }


def monitor_post_pull_contact(
    controller,
    target_link,
    *,
    steps=50,
    impulse_epsilon=1e-8,
):
    """
    每个 FlowBot 高层 pull 后检查夹爪是否仍与目标 link 保持物理接触。

    这里不要求 50 个 step 都双侧夹紧；操作过程中受力会导致左右接触
    交替变化。只要窗口内曾出现有效 finger-target contact，就不立即判滑脱。
    如果整个窗口左右手指都没有任何有效接触，则认为 grasp_lost。
    """
    steps = int(steps)
    frames = []

    for _ in range(steps):
        controller.keep_gripper_closed()
        controller.step()

        frames.append(
            read_twofinger_contact(
                controller.scene,
                controller.left_finger_link,
                controller.right_finger_link,
                target_link,
                impulse_epsilon=impulse_epsilon,
            )
        )

    any_count = sum(
        int(x["any_finger_contact"])
        for x in frames
    )
    bilateral_count = sum(
        int(x["bilateral_contact"])
        for x in frames
    )

    return {
        "grasp_lost": bool(any_count == 0),
        "steps": steps,
        "any_contact_frames": int(any_count),
        "any_contact_fraction": float(any_count / float(max(steps, 1))),
        "bilateral_frames": int(bilateral_count),
        "bilateral_fraction": float(
            bilateral_count / float(max(steps, 1))
        ),
        "final_frame": frames[-1] if frames else None,
    }


def monitor_target_engagement(
    controller,
    target_link,
    *,
    steps=50,
    impulse_epsilon=1e-8,
):
    """Push pre-operation engagement: closed hand/finger contact with target."""
    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps 必须 > 0")

    frames = []
    for _ in range(steps):
        controller.keep_gripper_closed()
        controller.step()
        finger = read_twofinger_contact(
            controller.scene,
            controller.left_finger_link,
            controller.right_finger_link,
            target_link,
            impulse_epsilon=impulse_epsilon,
        )
        hand = _contact_impulse_between(
            controller.scene,
            controller.hand_link,
            target_link,
            impulse_epsilon=impulse_epsilon,
        )
        frames.append(
            {
                **finger,
                "hand_contact": bool(hand["effective"]),
                "hand_total_impulse": float(hand["total_impulse"]),
                "target_engagement": bool(
                    finger["any_finger_contact"] or hand["effective"]
                ),
            }
        )

    engaged = sum(int(frame["target_engagement"]) for frame in frames)
    return {
        "engagement_success": bool(engaged > 0),
        "definition": (
            "closed gripper/hand has nonzero-impulse contact with target part "
            "before formal pushing motion"
        ),
        "steps": steps,
        "engaged_frames": int(engaged),
        "engaged_fraction": float(engaged / float(steps)),
        "final_frame": frames[-1],
    }
