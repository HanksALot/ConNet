"""
    model settings
"""

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoderVideo',
    pretrained=None,
    backbone=dict(
        type='Semi3dTeNet',
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=3,
        channels=64,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=2,
        threshold=0.3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[dict(type='CrossEntropyLoss', loss_weight=1.0),
                     dict(type='DiceLoss', loss_weight=4.0)]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
