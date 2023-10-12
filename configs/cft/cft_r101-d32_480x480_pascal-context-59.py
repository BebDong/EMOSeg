_base_ = ['../_base_/default_runtime.py']

# model
num_classes = 59
crop_size = (480, 480)
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
    ),
    decode_head=dict(
        type='CFTHead',
        feature_strides=(4, 8, 16, 32),
        num_heads=4,
        attn_drop_rate=0.,
        drop_rate=0.,
        qkv_bias=True,
        mlp_ratio=4,
        ln_norm_cfg=dict(type='LN', eps=1e-6),
        in_channels=(256, 512, 1024, 2048),
        channels=256,
        num_classes=num_classes,
        dropout_ratio=.1,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        in_index=(0, 1, 2, 3),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='MaskLoss', num_classes=num_classes, mask_weight=5.0, dice_weight=2.0, loss_weight=1.0)],
        init_cfg=[dict(type='Normal', std=0.01, override=dict(name='conv_seg')),
                  dict(type='TruncNormal', layer='Linear', std=0.02),
                  dict(type='Constant', layer='LayerNorm', val=1., bias=0.)]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)),
)

# dataset
dataset_type = 'PascalContextDataset59'
data_root = 'data/VOCdevkit/VOC2010'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RandomResize', scale=(520, 520), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(520, 520), keep_ratio=True),
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
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/train.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# schedule
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5),
    paramwise_cfg=dict(paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10., decay_mult=1000)}))
)
param_scheduler = [dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
                   dict(type='PolyLR', eta_min=0., power=1.0, begin=1500, end=80000, by_epoch=False)]
# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# runtime
visualizer = dict(type='SegLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')],
                  name='visualizer')  # debug w/o wandb
# visualizer = dict(type='SegLocalVisualizer',
#                   vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')],
#                   name='visualizer')
