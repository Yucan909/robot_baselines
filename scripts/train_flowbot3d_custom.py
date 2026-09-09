import argparse
import json
import time
from pathlib import Path

import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from flowbot3d.models.artflownet import (
    ArtFlowNet,
    ArtFlowNetParams,
    artflownet_loss,
)


HOME = Path.home()

TRAIN_ROOT = (
    HOME
    / "robot_baselines/data/flowbot3d_custom/full/train"
)

RUN_ROOT = (
    HOME
    / "robot_baselines/results/flowbot3d/training"
)


class FlowBotPTDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.pt"))

        if len(self.files) == 0:
            raise RuntimeError(
                f"没有找到训练数据: {self.root}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        raw = torch.load(
            self.files[idx],
            map_location="cpu",
            weights_only=False,
        )

        # 网络训练只需要这三个量
        return Data(
            pos=raw.pos,
            flow=raw.flow,
            mask=raw.mask,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--accumulate",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    pl.seed_everything(
        args.seed,
        workers=True,
    )

    # ========================================================
    # 1. 数据集
    # ========================================================

    dataset = FlowBotPTDataset(
        TRAIN_ROOT
    )

    print("=" * 80)
    print("FlowBot3D 自定义训练")
    print("=" * 80)

    print("训练数据目录:", TRAIN_ROOT)
    print("训练样本数:", len(dataset))
    print("Batch size:", args.batch_size)
    print("梯度累计:", args.accumulate)
    print(
        "有效 batch size:",
        args.batch_size * args.accumulate,
    )
    print("Epochs:", args.epochs)
    print("Learning rate:", args.lr)
    print("DataLoader workers:", args.workers)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(
            args.workers > 0
        ),
    )

    # ========================================================
    # 2. GPU 前向 + 反向预检查
    # ========================================================

    print()
    print("=" * 80)
    print("执行一个 batch 的 GPU 预检查")
    print("=" * 80)

    probe_batch = next(
        iter(loader)
    ).cuda()

    probe_model = ArtFlowNet(
        p=ArtFlowNetParams(
            mask_input_channel=True
        ),
        lr=args.lr,
    ).cuda()

    probe_model.train()

    pred = probe_model(
        probe_batch
    )

    n_nodes = torch.tensor(
        [
            d.num_nodes
            for d in probe_batch.to_data_list()
        ],
        dtype=torch.long,
        device="cuda",
    )

    probe_loss = artflownet_loss(
        pred,
        probe_batch.flow,
        n_nodes,
    )

    probe_loss.backward()

    print(
        "Pred shape:",
        tuple(pred.shape),
    )

    print(
        "GT shape:",
        tuple(probe_batch.flow.shape),
    )

    print(
        "预检查 Loss:",
        float(probe_loss.detach().cpu()),
    )

    print(
        "当前 GPU 显存:",
        round(
            torch.cuda.memory_allocated()
            / 1024**3,
            2,
        ),
        "GB",
    )

    if not torch.isfinite(
        probe_loss
    ):
        raise RuntimeError(
            "GPU 预检查出现非有限 loss"
        )

    del pred
    del probe_loss
    del probe_batch
    del probe_model

    torch.cuda.empty_cache()

    print("GPU 预检查通过")

    # ========================================================
    # 3. 正式 ArtFlowNet
    # ========================================================

    model = ArtFlowNet(
        p=ArtFlowNetParams(
            mask_input_channel=True
        ),
        lr=args.lr,
    )

    # ========================================================
    # 4. 实验目录
    # ========================================================

    run_name = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        RUN_ROOT / run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulate,
        "effective_batch_size": (
            args.batch_size
            * args.accumulate
        ),
        "learning_rate": args.lr,
        "workers": args.workers,
        "seed": args.seed,
        "mask_input_channel": True,
        "training_samples": len(dataset),
        "train_root": str(TRAIN_ROOT),
    }

    with open(
        run_dir / "config.json",
        "w",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    # ========================================================
    # 5. checkpoint
    # ========================================================

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(
            run_dir / "checkpoints"
        ),
        filename="epoch-{epoch:03d}",
        every_n_epochs=10,
        save_top_k=-1,
        save_last=True,
    )

    logger = CSVLogger(
        save_dir=str(run_dir),
        name="csv_logs",
    )

    # ========================================================
    # 6. Lightning Trainer
    # ========================================================

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate,
        logger=logger,
        callbacks=[
            checkpoint_callback
        ],
        log_every_n_steps=10,
        deterministic="warn",
        enable_progress_bar=True,
    )

    print()
    print("=" * 80)
    print("开始正式训练")
    print("=" * 80)

    print("Run directory:")
    print(run_dir)

    trainer.fit(
        model,
        train_dataloaders=loader,
    )

    # ========================================================
    # 7. 保存最终模型
    # ========================================================

    final_ckpt = (
        run_dir / "final.ckpt"
    )

    trainer.save_checkpoint(
        str(final_ckpt)
    )

    print()
    print("=" * 80)
    print("FlowBot3D 训练完成")
    print("=" * 80)

    print("最终 checkpoint:")
    print(final_ckpt)


if __name__ == "__main__":
    main()
