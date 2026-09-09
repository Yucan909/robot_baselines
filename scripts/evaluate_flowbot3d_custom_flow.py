import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from flowbot3d.models.artflownet import ArtFlowNet


HOME = Path.home()

VAL_ROOT = (
    HOME
    / "robot_baselines/data/flowbot3d_custom/full/val"
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
    )

    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)

    print("=" * 80)
    print("FlowBot3D 验证集三维运动场评估")
    print("=" * 80)

    print("模型:")
    print(checkpoint)

    print()
    print("加载模型...")

    model = ArtFlowNet.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
    )

    model = model.cuda()
    model.eval()

    files = sorted(
        VAL_ROOT.glob("*.pt")
    )

    print("验证样本数量:", len(files))

    if len(files) == 0:
        raise RuntimeError(
            f"没有找到验证数据: {VAL_ROOT}"
        )

    results = []

    category_results = defaultdict(list)

    for i, path in enumerate(files):

        d = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        mask = d.mask.bool()

        if mask.sum() == 0:
            print(
                "警告：没有目标活动点:",
                path.name,
            )
            continue

        with torch.no_grad():

            pred = model.predict(
                d.pos,
                d.mask,
            )

            pred = pred.cpu()

        gt = d.flow

        pred_target = pred[mask]
        gt_target = gt[mask]

        # ------------------------------------------
        # 1. 平均欧氏距离误差
        # ------------------------------------------

        rmse = (
            pred_target
            - gt_target
        ).norm(
            dim=1
        ).mean()

        # ------------------------------------------
        # 2. 方向余弦相似度
        # ------------------------------------------

        cosine = (
            F.cosine_similarity(
                pred_target,
                gt_target,
                dim=1,
            )
            .mean()
        )

        # ------------------------------------------
        # 3. 运动幅值误差
        # ------------------------------------------

        magnitude_error = (
            pred_target.norm(
                dim=1
            )
            - gt_target.norm(
                dim=1
            )
        ).abs().mean()

        result = {
            "file": path.name,
            "shape_id": str(
                d.shape_id
            ),
            "link_name": str(
                d.link_name
            ),
            "category": str(
                d.category
            ),
            "target_points": int(
                mask.sum()
            ),
            "rmse": float(
                rmse
            ),
            "cosine_similarity": float(
                cosine
            ),
            "magnitude_error": float(
                magnitude_error
            ),
        }

        results.append(
            result
        )

        category_results[
            str(d.category)
        ].append(result)

        print(
            f"[{i+1:3d}/{len(files):3d}] "
            f"{d.shape_id}/{d.link_name} "
            f"方向相似度={float(cosine):.4f} "
            f"距离误差={float(rmse):.4f}"
        )

    if len(results) == 0:
        raise RuntimeError(
            "没有得到有效验证结果"
        )

    # ======================================================
    # 整体平均
    # ======================================================

    mean_rmse = sum(
        x["rmse"]
        for x in results
    ) / len(results)

    mean_cosine = sum(
        x["cosine_similarity"]
        for x in results
    ) / len(results)

    mean_magnitude_error = sum(
        x["magnitude_error"]
        for x in results
    ) / len(results)

    # ======================================================
    # 类别平均
    # ======================================================

    category_summary = {}

    for category, items in category_results.items():

        category_summary[
            category
        ] = {
            "num_targets": len(
                items
            ),
            "mean_rmse": sum(
                x["rmse"]
                for x in items
            ) / len(items),
            "mean_cosine_similarity": sum(
                x["cosine_similarity"]
                for x in items
            ) / len(items),
            "mean_magnitude_error": sum(
                x["magnitude_error"]
                for x in items
            ) / len(items),
        }

    summary = {

        "checkpoint": str(
            checkpoint
        ),

        "num_validation_targets": len(
            results
        ),

        "overall": {

            "mean_rmse":
                mean_rmse,

            "mean_cosine_similarity":
                mean_cosine,

            "mean_magnitude_error":
                mean_magnitude_error,
        },

        "categories":
            category_summary,

        "targets":
            results,
    }

    output = (
        checkpoint.parent
        / "validation_flow_metrics.json"
    )

    with open(
        output,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("验证集整体结果")
    print("=" * 80)

    print(
        "有效验证目标:",
        len(results),
    )

    print(
        "平均距离误差:",
        mean_rmse,
    )

    print(
        "平均方向余弦相似度:",
        mean_cosine,
    )

    print(
        "平均幅值误差:",
        mean_magnitude_error,
    )

    print()
    print("结果保存:")
    print(output)

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
