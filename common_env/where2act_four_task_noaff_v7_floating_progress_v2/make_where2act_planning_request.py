import argparse
import json
from pathlib import Path

import numpy as np


def main(args):

    result_path = (
        Path(args.result)
        .expanduser()
        .resolve()
    )

    if not result_path.exists():
        raise FileNotFoundError(
            result_path
        )

    with result_path.open() as f:
        result = json.load(f)

    contact = np.asarray(
        result[
            "interaction_point_world"
        ],
        dtype=np.float64,
    ).reshape(3)

    T_w2a = np.asarray(
        result[
            "grasp_pose_world"
        ],
        dtype=np.float64,
    )

    if T_w2a.shape != (4, 4):
        raise RuntimeError(
            "grasp_pose_world不是4x4: "
            f"{T_w2a.shape}"
        )

    R = T_w2a[
        :3,
        :3
    ].copy()

    det = float(
        np.linalg.det(R)
    )

    orth_err = float(
        np.linalg.norm(
            R.T @ R
            - np.eye(3)
        )
    )

    if abs(det - 1.0) > 1e-4:
        raise RuntimeError(
            f"rotation det异常: {det}"
        )

    if orth_err > 1e-4:
        raise RuntimeError(
            "rotation非正交: "
            f"{orth_err}"
        )

    # --------------------------------------------------------
    # Where2Act最终用于gripper root的旋转：
    #
    #   R = [forward, left, up]
    #
    # 第三列即从gripper指向物体的接近方向。
    #
    # 我们的panda_grasptarget同样约定：
    #   +Z = approach axis
    # --------------------------------------------------------

    approach = R[
        :,
        2
    ].copy()

    approach /= (
        np.linalg.norm(
            approach
        )
        + 1e-12
    )

    w2a_root = T_w2a[
        :3,
        3
    ]

    root_to_contact = (
        contact
        - w2a_root
    )

    axial_offset = float(
        np.dot(
            root_to_contact,
            approach,
        )
    )

    lateral_offset = float(
        np.linalg.norm(
            root_to_contact
            - axial_offset
            * approach
        )
    )

    # 这是硬校验。
    # 当前Where2Act wrapper设计应让root位于contact后方约10cm。
    if axial_offset <= 0:
        raise RuntimeError(
            "Where2Act approach方向反了："
            f"axial_offset={axial_offset}"
        )

    if lateral_offset > 1e-4:
        raise RuntimeError(
            "Where2Act root/contact "
            "不沿approach轴："
            f"{lateral_offset}"
        )

    # --------------------------------------------------------
    # 完整Panda要规划的是grasptarget，而不是W2A floating
    # gripper的root。
    # --------------------------------------------------------

    T_pregrasp = np.eye(
        4,
        dtype=np.float64,
    )

    T_pregrasp[
        :3,
        :3
    ] = R

    T_pregrasp[
        :3,
        3
    ] = (
        contact
        - float(
            args.pregrasp_distance
        )
        * approach
    )

    # 后续SAPIEN最终接近使用。
    T_contact = np.eye(
        4,
        dtype=np.float64,
    )

    T_contact[
        :3,
        :3
    ] = R

    T_contact[
        :3,
        3
    ] = (
        contact
        - float(
            args.contact_standoff
        )
        * approach
    )

    request = {
        "version":
            1,

        "shape_id":
            str(
                result[
                    "shape_id"
                ]
            ),

        "target_link":
            str(
                result[
                    "target_link"
                ]
            ),

        "trial_seed":
            int(
                result[
                    "trial_seed"
                ]
            ),

        "initial_ratio":
            float(
                result[
                    "initial_ratio"
                ]
            ),

        "network_trained":
            bool(
                result.get(
                    "trained",
                    False,
                )
            ),

        "goal_frame":
            "panda_grasptarget",

        "interaction_point_world":
            contact.tolist(),

        "approach_axis_world":
            approach.tolist(),

        "w2a_gripper_root_pose_world":
            T_w2a.tolist(),

        "pregrasp_pose_world":
            T_pregrasp.tolist(),

        "contact_pose_world":
            T_contact.tolist(),

        "pregrasp_distance":
            float(
                args.pregrasp_distance
            ),

        "contact_standoff":
            float(
                args.contact_standoff
            ),

        "diagnostics": {
            "rotation_det":
                det,

            "rotation_orthogonality_error":
                orth_err,

            "w2a_root_to_contact_axial":
                axial_offset,

            "w2a_root_to_contact_lateral":
                lateral_offset,
        },
    }

    output = (
        Path(args.output)
        .expanduser()
        .resolve()
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w"
    ) as f:

        json.dump(
            request,
            f,
            indent=2,
        )

    print("=" * 100)
    print("WHERE2ACT -> PANDA PLANNING REQUEST")
    print("=" * 100)

    print(
        "shape:",
        request[
            "shape_id"
        ],
    )

    print(
        "target:",
        request[
            "target_link"
        ],
    )

    print(
        "network trained:",
        request[
            "network_trained"
        ],
    )

    print(
        "contact:",
        contact,
    )

    print(
        "approach axis:",
        approach,
    )

    print(
        "W2A root -> contact axial:",
        axial_offset,
    )

    print(
        "W2A root -> contact lateral:",
        lateral_offset,
    )

    print(
        "pregrasp:",
        T_pregrasp[
            :3,
            3
        ],
    )

    print(
        "contact target:",
        T_contact[
            :3,
            3
        ],
    )

    print(
        "saved:",
        output,
    )

    print()
    print(
        "WHERE2ACT -> PANDA REQUEST: PASS"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--pregrasp-distance",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--contact-standoff",
        type=float,
        default=0.005,
    )

    main(
        parser.parse_args()
    )
