import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import UPSAMPLE_LAYERS, build_activation_layer, build_norm_layer


@UPSAMPLE_LAYERS.register_module()
class DeconvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False,
                 norm_cfg=None,
                 act_cfg=None,
                 *,
                 kernel_size=4,
                 scale_factor=2):
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        super(DeconvModule, self).__init__()

        assert (kernel_size - scale_factor >= 0) and (kernel_size - scale_factor) % 2 == 0, \
            f'kernel_size should be greater than or equal to scale_factor ' \
            f'and (kernel_size - scale_factor) should be even numbers, ' \
            f'while the kernel size is {kernel_size} and scale_factor is ' \
            f'{scale_factor}.'

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        self.with_cp = with_cp
        deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        activate = build_activation_layer(act_cfg)
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.deconv_upsamping, x)
        else:
            out = self.deconv_upsamping(x)
        return out
