# 清理记录

## 保留原则

- 每个 baseline 保留一份官方源码快照，并记录上游 URL 与提交。
- 保留最终复现路径、固定配置、最终选定 checkpoint、有效原始/紧凑数据和必要依赖。
- 可再生成的大缓存只保留紧凑来源与重建脚本。
- 不保留历史 rollout、视频、逐阶段审计包、重复 checkpoint、控制台日志、Python 缓存、编译目录、损坏或未完成下载。

## 已排除的关键内容

- `robot_baselines/results/pa3ff/padp_data_v4_timeindexed_baseframe_fourtask`：约 92 GB 的可再生成特征缓存。
- `robot_baselines/results` 中其余历史评测/训练输出：只抽取最终必要权重、索引和冻结元数据。
- `下载/a` 中四任务数据已迁入 `data/where2act_four_task`；仅下载缓存、运行日志、临时文件和错误布局 smoke 目录被排除。
- `下载/ArticuBot/data/dataset.sparse.zip`：结构索引存在，但压缩数据为空洞，解压从首项开始报错。
- `下载/ArticuBot/data/diverse_objects.sparse.zip`：同为损坏稀疏占位包。
- `下载/ArticuBot/data/diverse_objects.zip*.part`：未完成分片。
- 下载目录内所有 `stage*`、`patch*`、`dev_v*`、`smoke*`、`audit*`、旧结果包和重复压缩包；最终有效代码已并入四个 baseline 目录。

## 实际清理结果

- 新整理目录已替换为正式 `/home/feng/robot_baselines`，顶层源码只保留四个 baseline。
- 原工作区、历史输出和散落脚本已进入专用清理批次；下载目录共移出 177 个 baseline 相关项目。有效数据已并入正式包，约 13 GB 的重复归档、缓存和脚本进入清理批次。
- 专用清理批次合计约 149 GB。正式目录完成数据校验、Git 校验和语法校验后，该批次永久删除，不再可恢复。
