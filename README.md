# 四个机器人基线复现包

这是清理后的单仓库工作区，只保留 FlowBot3D、Where2Act、PA3FF 和 ArticuBot 的官方源码、最终复现适配、必要依赖、有效数据与最终选定权重。历史实验结果、调试版本、重复 checkpoint、`__pycache__`、编译产物和损坏下载均不属于本仓库。

## 目录

- `repos/flowbot3d`：FlowBot3D 官方源码（上游提交见 `manifests/UPSTREAM_SOURCES.tsv`）。
- `repos/where2act`：Where2Act 官方源码，加四任务 V7 和最终复现代码。
- `repos/pa3ff_official`：PA3FF 官方源码、RTX 5080 兼容修改、Sonata 权重。
- `repos/pa3ff_official/reproduction`：精简后的 PA3FF/PADP V5 训练与正式评测闭包。
- `repos/articubot`：ArticuBot 官方源码、官方权重和最终适配器。
- `data`：本机可直接使用的数据；大数据本身不直接进入普通 Git 历史。
- `dataset_archives`：由 Git LFS 跟踪的分卷数据归档。
- `results`：仅保留运行所需的最终 checkpoint、训练索引与冻结元数据；不是历史结果归档。
- `common_env`、`configs`、`deps`、`splits`：四个基线共用的冻结协议、配置和依赖。

## 数据边界

详细清单见 `DATASETS.md`。本机现有的完整 PartNet-Mobility、FlowBot3D 缓存、Where2Act 四任务数据、PA3FF 紧凑训练源，以及 ArticuBot 已成功落盘的对象子集和官方权重都保留了。ArticuBot 的两个 `*.sparse.zip` 经解压测试确认损坏，另一个 `.part` 只是未完成下载，因此没有混入最终包。

PA3FF 的约 92 GB `*_pa3ff_field_f16.npy` 是可由紧凑源重新生成的特征缓存，不是原始数据；它已从最终包排除。重新生成顺序为：

```bash
conda activate pa3ff
cd /home/feng/robot_baselines/repos/pa3ff_official/reproduction
python build_padp_v4_timeindexed_baseframe.py
python precompute_padp_v4_geometry.py
python precompute_pa3ff_v4_features.py
python audit_pa3ff_v4_feature_cache.py
```

## 校验

```bash
cd /home/feng/robot_baselines
python scripts/verify_bundle.py
```

## Git / Git LFS

仓库已配置 Git LFS。克隆后先执行：

```bash
git lfs install
git lfs pull
bash scripts/restore_data_archives.sh
python scripts/verify_bundle.py
```

推送前需确认远端账户有足够的 LFS 存储/带宽；数据分卷并不会绕过总容量配额。当前没有擅自绑定或推送到任何远端。
