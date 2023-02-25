exp_name = 'test_config'

model = dict(
    type='TestNet',
    body=dict(
        type='TestModel'
    ),
    mse_loss=dict(type='MSELoss', loss_weight=1.0),
)

# model training and testing settings
train_cfg = None
test_cfg = None

# dataset settings
train_dataset_type = 'TripleDataset'
val_dataset_type = 'TripleDataset'
test_dataset_type = 'SRFolderLRDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=320),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path']),
    dict(type='ImageToTensor', keys=['lq'])
]

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=train_dataset_type,
        lq_folder='/datafile/lcf2021/uw_train/input_train',
        gt_folder='/datafile/lcf2021/uw_train/gt_train',
        pipeline=train_pipeline,
        scale=1,
        filename_tmpl='{}'),
    val=dict(
        type=val_dataset_type,
        lq_folder='/datafile/lcf2021/uw_test/Test-R90/raw',
        gt_folder='/datafile/lcf2021/uw_test/Test-R90/ref',
        pipeline=val_pipeline,
        scale=1,
        filename_tmpl='{}'),
    test=dict(
        type=test_dataset_type,
        lq_folder='/datafile/lcf2021/uw_test/Test-C60',
        pipeline=test_pipeline,
        scale=1,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(type='Adam', lr=1e-3, betas=(0.9, 0.999))

# learning policy
total_iters = 100000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[50000],
    gamma=0.5)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# 'no dist' do not have 'gpu_collect'
# 'dist' do
evaluation = dict(interval=500, save_image=True)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='WandbLoggerHook', init_kwargs=dict(project='wb')),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
save_dir = f'./work_dirs/{exp_name}/results'
load_from = None
resume_from = None
workflow = [('train', 1)]