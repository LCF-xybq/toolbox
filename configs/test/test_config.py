exp_name = 'test_config'

model = dict(
    type='Test',
    body=dict(
        type='TestNet'
    ),
    mse_loss=dict(type='MSELoss', loss_weight=1.0),
    perc_loss=dict(
        type='PerceptualLoss',
        layer_weights={'34': 1.0},
        vgg_type='vgg19',
        perceptual_weight=0.1,
        style_weight=0,
        norm_img=False
    )
)

# model training and testing settings
train_cfg = None
test_cfg = None

# dataset settings
dataset_type = 'PairedImageDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='input',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        channel_order='rgb'),
    dict(type='Resize', scale=(280, 280),keys=['input', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['input', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Collect', keys=['input', 'gt'], meta_keys=['input_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['input', 'gt'])
]

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='input',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        channel_order='rgb'),
    dict(type='Collect', keys=['input', 'gt'], meta_keys=['input_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['input', 'gt'])
]


data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        root=r'D:\Program_self\Datasets\UIEB',
        prefix=dict(img='raw-890', gt='reference-890'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        root=r'E:\dataset\uw_test\Test-R90',
        prefix=dict(img='raw', gt='ref'),
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        root=r'D:\Program_self\Datasets\UIEB',
        prefix=dict(img='challenging-60', gt='challenging-60'),
        pipeline=val_pipeline)
)

# optimizer
optimizers = dict(type='Adam', lr=1e-3, betas=(0.9, 0.999))

# learning policy
total_iters = 100

checkpoint_config = dict(interval=5, save_optimizer=True, by_epoch=False)
# 'no dist' do not have 'gpu_collect'
# 'dist' do
evaluation = dict(interval=5, save_image=True)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='WandbLoggerHook', init_kwargs=dict(project='wb')),
    ])

lr_config = dict(policy='Fixed', by_epoch=False)

visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
save_dir = f'./work_dirs/{exp_name}/results'
load_from = None
resume_from = None
workflow = [('train', 1)]
