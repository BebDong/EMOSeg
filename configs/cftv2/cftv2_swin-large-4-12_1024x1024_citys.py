_base_ = ['../_base_/default_runtime.py']

# model
num_classes = 19
crop_size = (1024, 1024)
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        pretrain_img_size=384,
        embed_dims=192,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
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
        type='CFTHeadV2',
        num_heads=4,
        attn_drop_rate=0.,
        drop_rate=0.,
        qkv_bias=True,
        mlp_ratio=4,
        ln_norm_cfg=dict(type='LN', eps=1e-6),
        use_memory=True,
        momentum_cfg=dict(start=0.1, use_poly=False, total_iter=160000, power=0.9, eta_min=0.009),
        init_memory=None,
        in_channels=(192, 384, 768, 1536),
        channels=256,
        num_classes=num_classes,
        dropout_ratio=.1,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        in_index=(0, 1, 2, 3),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_mask_decode=dict(type='MaskLoss', mask_weight=5.0, dice_weight=2.0, loss_weight=1.0),
        init_cfg=[dict(type='Normal', std=0.01, override=dict(name='conv_seg')),
                  dict(type='TruncNormal', layer='Linear', std=0.02),
                  dict(type='Constant', layer='LayerNorm', val=1., bias=0.)]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))
)

# dataset
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
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
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# schedule
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW', lr=4e-5, betas=(0.9, 0.999), weight_decay=0.03),
                     paramwise_cfg=dict(
                         custom_keys={
                             'absolute_pos_embed': dict(decay_mult=0.),
                             'relative_position_bias_table': dict(decay_mult=0.),
                             'norm': dict(decay_mult=0.),
                             'head': dict(lr_mult=10.)}))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=160000, by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000))
custom_hooks = [dict(type='RunnerInfoHook', priority='NORMAL')]

# runtime
visualizer = dict(type='SegLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')],
                  name='visualizer')
