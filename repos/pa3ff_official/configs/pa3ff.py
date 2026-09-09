_base_ = ["./default_runtime.py"]

# misc custom setting
batch_size = 1  # dummy value
num_worker = 1  # dummy value
mix_prob = 0  # dummy value
empty_cache = False
enable_amp = True

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]

# model settings
model = dict(
    type="PA3FF",
    backbone_dim=1088,
    output_dim=768,
    freeze_backbone=True,
    max_grouping_scale=2,
    use_hierarchy_losses=True,
    backbone=None
)

# scheduler settings
epoch = 10000
eval_epoch = 1000
lr = 1e-6
optimizer = dict(type="AdamW", lr=lr, weight_decay=1e-4)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[lr],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=25.0,
)

dataset_type = "PartDataset"

# eval
val_scales_list = [0.0, 0.5, 1.0, 1.5, 2.0]
mesh_voting = False

data = dict(
    train=dict(
        type=dataset_type,
        category="bottle",
        data_root="./partnet/train",
        random_catogory=False,
        transform = [
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
    ),
    val=dict(
        type=dataset_type,
        category="bottle",
        data_root="./partnet/val",
        random_catogory=False,
        transform = [
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
    )
)