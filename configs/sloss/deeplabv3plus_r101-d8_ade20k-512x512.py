_base_ = ['../_base_/default_runtime.py']

# model settings
num_classes = 150
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    size=crop_size,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        # sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=dict(
            # type='SensitiveLoss', num_classes=num_classes, gamma=0.3, use_scale=True, loss_weight=1.0
            # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            class_weight=[0.7647, 0.7792, 0.7868, 0.8013, 0.8115, 0.8145, 0.8198, 0.8443, 0.8507, 0.8551,
                          0.8553, 0.859, 0.8614, 0.8642, 0.8756, 0.8791, 0.8798, 0.8835, 0.882, 0.8826,
                          0.8843, 0.8995, 0.9037, 0.9053, 0.9087, 0.9098, 0.9151, 0.9174, 0.9257, 0.9262,
                          0.9262, 0.9257, 0.9437, 0.9456, 0.95, 0.9491, 0.9578, 0.9612, 0.9628, 0.9643,
                          0.9633, 0.9673, 0.9687, 0.9719, 0.9751, 0.9765, 0.9779, 0.9776, 0.9797, 0.9789,
                          0.9796, 0.9773, 0.98, 0.9837, 0.9868, 0.983, 0.9831, 0.9852, 0.9882, 0.9874,
                          0.9881, 0.9892, 0.9909, 0.9987, 0.9937, 0.9947, 0.9926, 0.998, 0.9977, 1.0,
                          1.0042, 1.0065, 1.0033, 1.0044, 1.0157, 1.0133, 1.0182, 1.0269, 1.0216, 1.0254,
                          1.0258, 1.028, 1.032, 1.0287, 1.0329, 1.036, 1.0356, 1.038, 1.0366, 1.0409, 1.0384,
                          1.039, 1.0414, 1.0474, 1.0443, 1.0468, 1.0493, 1.0488, 1.0484, 1.0536, 1.05, 1.0485,
                          1.0497, 1.0569, 1.0608, 1.0587, 1.0619, 1.066, 1.0576, 1.0609, 1.0613, 1.0578, 1.0668,
                          1.0659, 1.067, 1.0719, 1.0667, 1.0674, 1.0663, 1.0648, 1.0814, 1.0759, 1.0843, 1.0826,
                          1.0914, 1.0845, 1.0932, 1.081, 1.0908, 1.0841, 1.0914, 1.0938, 1.087, 1.0901, 1.0916,
                          1.0905, 1.0945, 1.0936, 1.1, 1.0992, 1.0946, 1.098, 1.1033, 1.1054, 1.1024, 1.0972,
                          1.1167, 1.13, 1.1257, 1.1397]
            # type='FocalLoss2d', gamma=2.0, loss_weight=1.0
            # type='LovaszLoss', loss_weight=1.0, per_image=False, reduction='none'
            # type='TverskyLoss', loss_weight=1.0
            # type='PolyLoss', epsilon=2.0, loss_weight=1.0,
        )
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset
dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/validation', seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# schedule
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
param_scheduler = [dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end=160000, by_epoch=False)]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# runtime
visualizer = dict(type='SegLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')],
                  name='visualizer')  # debug w/o wandb
