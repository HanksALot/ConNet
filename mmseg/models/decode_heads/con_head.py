import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.models.utils import resize
from ..builder import HEADS
from .base_decode_head import BaseDecodeHead


@HEADS.register_module()
class ConHead(BaseDecodeHead):
    def __init__(self,
                 head_flag,
                 freeze=False,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 norm_eval=False,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(ConHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        self.norm_eval = norm_eval
        if head_flag is None:
            raise IOError('head_flag is None')
        self.head_flag = head_flag
        self.freeze = freeze

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        for i in range(num_convs):
            _in_channels = self.in_channels if i == 0 else self.channels
            convs.append(
                ConvModule(
                    _in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        if len(convs) == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            inputs=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        return loss

    def check_freeze_status(self):
        if all(not param.requires_grad for param in self.parameters()):
            print("ConHead is fully frozen.")
        else:
            print("ConHead is not fully frozen.")

    def _freeze_stages(self):
        """Freeze all layers in ConHead by setting requires_grad=False."""
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if self.freeze is True and self.head_flag == 'cdt':
            self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
