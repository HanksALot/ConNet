import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import UPSAMPLE_LAYERS, ConvModule

from mmseg.models.utils import Upsample


@UPSAMPLE_LAYERS.register_module()
class InterpConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False,
                 norm_cfg=None,
                 act_cfg=None,
                 *,
                 conv_cfg=None,
                 conv_first=False,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 upsample_cfg=None):
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        if upsample_cfg is None:
            upsample_cfg = dict(
                scale_factor=2, mode='bilinear', align_corners=False)
        super(InterpConv, self).__init__()

        self.with_cp = with_cp
        conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        upsample = Upsample(**upsample_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.interp_upsample, x)
        else:
            out = self.interp_upsample(x)
        return out
