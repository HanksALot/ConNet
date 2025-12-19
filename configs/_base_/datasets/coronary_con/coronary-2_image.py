"""
    dataset settings
"""

dataset_type = 'CoronaryDCtlDataset'
data_root = '../../data/convert_data/coronary-2_image'
img_norm_cfg = dict(
    mean=[101.1323], std=[31.3953], to_rgb=False)
img_scale = (512, 512)
test_pipeline = [
    dict(type='LoadImageFromFileDCtl', color_type='unchanged'),
    dict(
        type='MultiScaleFlipAugDCtl',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='ResizeDCtl', keep_ratio=True),
            dict(type='RandomFlipDCtl'),
            dict(type='NormalizeDCtl', **img_norm_cfg),
            dict(type='ImageToTensorDCtl', keys=['img']),
            dict(type='CollectDCtl', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='seq_dir/test',
        ann_dir='ann_dir/test',
        pipeline=test_pipeline,
        frame_num=3,
        cdt_guidance=False))
