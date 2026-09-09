# 数据与权重清单

| 内容 | 本机路径 | 类型 | 说明 |
| --- | --- | --- | --- |
| PartNet-Mobility | `data/partnet-mobility` | 原始对象数据 | 四个基线共享；约 9.1 GB、约 41 万文件。 |
| FlowBot3D custom | `data/flowbot3d_custom` | 可再生成训练缓存 | 由 PartNet-Mobility 和固定 split 生成；约 811 MB。 |
| FlowBot3D pose/split source | `data/flowbot3d_pose_and_split_package.zip` | 原始姿态与划分元数据 | 保留一份经完整性测试的来源包；重复副本已排除。 |
| Where2Act four-task | `data/where2act_four_task` | 四任务原始/预处理轨迹数据 | 门和抽屉开/关四任务；约 27 GB、约 10.5 万文件。运行日志、临时文件及下载缓存已排除。 |
| PA3FF V3 compact source | `results/pa3ff/padp_data_v3_initial_state_aligned_fourtask` | 紧凑训练源 | 约 75 MB；用于重建被排除的 92 GB V4 特征缓存。 |
| ArticuBot parsed objects | `repos/articubot/data/dataset` | 已成功落盘的官方格式对象子集 | 约 105 MB；同时可访问共享 PartNet-Mobility。 |
| ArticuBot diverse objects | `repos/articubot/data/diverse_objects` | 已成功落盘的评测子集 | 当前约 1 MB。 |
| ArticuBot high-level weight | `repos/articubot/data/high_level_200_obj_ckpt.pth` | 官方模型权重 | 约 33 MB。 |
| ArticuBot low-level weight | `repos/articubot/data/low-level-ckpt` | 官方模型权重 | 约 4.2 GB。 |
| FlowBot3D selected weight | `results/flowbot3d/final/model.ckpt` | 最终复现权重 | 约 17 MB。 |
| Where2Act V7 weights | `repos/where2act/logs/four_task_train_v7_noaff_schema_robust/20260905_032913` | 四任务最终权重 | 每任务保留 critic/joint 的 `best-network.pth`。 |
| PA3FF/PADP V5 weight | `results/pa3ff/reproduction_v5/training_30000/checkpoints/step020000.pt` | DEV 选定权重 | SHA256 由冻结选择文件和 `manifests/SHA256SUMS` 双重记录。 |

ArticuBot README 中提到的完整 `dp3_demo` / `dp3_demo_combined_2_step_0` 训练演示在清理前并未完整存在于本机。最终包不会把损坏稀疏 ZIP 或未完成 `.part` 冒充成完整数据集；如以后补下官方演示数据，应放到 `repos/articubot/data`，重新运行归档脚本并更新清单。

`dataset_archives` 的每个归档可能被拆成多个 `part-*`。恢复脚本按顺序流式拼接并解压，不会在中间再生成一个重复的大归档文件。
