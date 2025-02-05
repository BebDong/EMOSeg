_base_ = ['../_base_/default_runtime.py']

# model settings
num_classes = 59
crop_size = (480, 480)
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
checkpoint_file = ('https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/'
                   'swin_small_patch4_window7_224_20220317-7ba6d6dd.pth')
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
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='SensitiveLoss', num_classes=num_classes, gamma=0.2, use_scale=True, loss_weight=1.0
            # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            # class_weight=[0.9752, 1.1304, 1.0394, 0.979, 1.1678, 0.9746, 0.9684, 0.9843, 1.0779, 1.0104,
            #               0.8645, 0.9524, 0.9868, 0.9177, 0.8935, 0.9964, 0.9434, 0.9809, 1.1404, 0.9986,
            #               1.1305, 1.013, 0.9012, 1.004, 0.9556, 0.9, 1.0835, 1.1341, 0.8632, 0.8645, 0.9675,
            #               1.1404, 1.1137, 0.9552, 0.9701, 1.4059, 0.8564, 1.1347, 1.0534, 0.9957, 0.9114,
            #               1.0241, 0.9884, 1.0245, 1.0236, 1.108, 0.8488, 1.0122, 0.9343, 0.9747, 1.0404,
            #               0.9482, 0.8689, 1.1267, 0.9776, 0.864, 0.903, 0.9728, 1.0239]
            # type='FocalLoss2d', gamma=2.0, loss_weight=1.0
        )
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
dataset_type = 'PascalContextDataset59'
data_root = 'data/VOCdevkit/VOC2010/'
img_scale = (520, 520)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RandomResize', scale=img_scale, ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
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
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
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
        data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# schedule
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
param_scheduler = [dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
                   dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=80000, by_epoch=False)]
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
