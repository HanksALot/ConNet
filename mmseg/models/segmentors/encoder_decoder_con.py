import cv2
import numpy as np
import torch
import torch.nn as nn

from mmseg.core import add_prefix
from mmseg.models.utils import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class EncoderDecoderCon(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 with_seg=True,
                 with_cdt=False,
                 with_con=False,
                 with_aux_cdt=False,
                 with_aux_con=False,
                 num_connect=0,
                 with_postpro=False,
                 with_ensemble=(False, 'add'),
                 ensemble_weight=(0.4, 0.3, 0.3)):
        super(EncoderDecoderCon, self).__init__(init_cfg)

        self.with_seg = with_seg
        self.with_cdt = with_cdt
        self.with_con = with_con
        self.with_aux_cdt = with_aux_cdt
        self.with_aux_con = with_aux_con
        self.num_connect = num_connect
        self.with_postpro = with_postpro
        self.with_ensemble = with_ensemble
        self.ensemble_weight = ensemble_weight
        self.size_vessel = [100, 30]

        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head
        assert backbone['num_iters'] == 0 if auxiliary_head is None \
            else len(auxiliary_head) == backbone['num_iters']
        assert isinstance(with_ensemble, tuple)
        assert isinstance(ensemble_weight, tuple)

    @property
    def with_decode_head(self):
        """bool: whether the segmentor has decode head"""
        return (hasattr(self, 'decode_head_seg') and self.decode_head_seg is not None) or \
               (hasattr(self, 'decode_head_cdt') and self.decode_head_cdt is not None) or \
               (hasattr(self, 'decode_head_con') and self.decode_head_con is not None)

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return (hasattr(self, 'auxiliary_head_seg') and self.auxiliary_head_seg is not None) or \
               (hasattr(self, 'auxiliary_head_cdt') and self.auxiliary_head_cdt is not None) or \
               (hasattr(self, 'auxiliary_head_con') and self.auxiliary_head_con is not None)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.align_corners = decode_head['decode_head_seg']['align_corners']
        if self.with_seg:
            self.decode_head_seg = builder.build_head(decode_head['decode_head_seg'])
            self.num_classes_seg = self.decode_head_seg.num_classes
            self.out_channels_seg = self.decode_head_seg.out_channels

        if self.with_cdt:
            self.decode_head_cdt = builder.build_head(decode_head['decode_head_cdt'])
            self.num_classes_cdt = self.decode_head_cdt.num_classes
            self.out_channels_cdt = self.decode_head_cdt.out_channels

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                if self.with_seg:
                    self.auxiliary_head_seg = nn.ModuleList()
                if self.with_aux_cdt:
                    self.auxiliary_head_cdt = nn.ModuleList()
                if self.with_aux_con:
                    self.auxiliary_head_con = nn.ModuleList()

                for aux_head_cfg in auxiliary_head:
                    if self.with_seg:
                        self.auxiliary_head_seg.append(
                            builder.build_head(aux_head_cfg['decode_head_seg']))
                    if self.with_aux_cdt:
                        self.auxiliary_head_cdt.append(
                            builder.build_head(aux_head_cfg['decode_head_cdt']))
                    if self.with_aux_con:
                        sub_auxiliary_head_con = nn.ModuleList()
                        for _ in range(self.num_connect):
                            sub_auxiliary_head_con.append(
                                builder.build_head(aux_head_cfg['decode_head_con']))
                        self.auxiliary_head_con.append(sub_auxiliary_head_con)
            else:
                raise NotImplementedError

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode_resize(self, output, img_size):
        if isinstance(output, list):
            return [resize(tensor, size=img_size, mode='bilinear',
                           align_corners=self.align_corners) for tensor in output]
        else:
            return resize(output, size=img_size, mode='bilinear',
                          align_corners=self.align_corners)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)

        seg_cdt_con_out = self._decode_head_forward_test(x, img_metas)

        seg_cdt_con_out = [self.encode_decode_resize(sub_out, img[0].shape[2:])
                           if sub_out is not None else None for sub_out in seg_cdt_con_out]

        return seg_cdt_con_out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, gt_semantic_cdt, gt_semantic_con):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        if self.with_seg:
            loss = self.decode_head_seg.forward_train(x, img_metas, gt_semantic_seg[-1], self.train_cfg)
            losses.update(add_prefix(loss, prefix='decode_seg'))
        if self.with_cdt:
            loss = self.decode_head_cdt.forward_train(x, img_metas, gt_semantic_cdt[-1], self.train_cfg)
            losses.update(add_prefix(loss, prefix='decode_cdt'))

        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate logits for decode head in inference."""

        seg_logits, cdt_logits = None, None
        if self.with_seg:
            seg_logits = self.decode_head_seg.forward_test(x, img_metas, self.test_cfg)

        if self.with_cdt:
            cdt_logits = self.decode_head_cdt.forward_test(x, img_metas, self.test_cfg)

        return [seg_logits, cdt_logits]

    def _auxiliary_head_forward_train(self, x, img_metas,
                                      gt_semantic_seg, gt_semantic_cdt, gt_semantic_con):
        """Run forward function and calculate loss for auxiliary head in
        training."""

        def update_losses(head_list, b_x, gt_list, prefix):
            if not isinstance(head_list, nn.ModuleList):
                raise NotImplementedError
            for idx, aux_head in enumerate(head_list):
                if isinstance(aux_head, nn.ModuleList):
                    for sub_idx in range(len(aux_head)):
                        loss = aux_head[sub_idx].forward_train(
                            b_x, img_metas, gt_list[sub_idx], self.train_cfg)
                        losses.update(add_prefix(loss, f'{prefix}_{idx}_{sub_idx}'))
                else:
                    loss = aux_head.forward_train(b_x, img_metas, gt_list, self.train_cfg)
                    losses.update(add_prefix(loss, f'{prefix}_{idx}'))

        losses = dict()
        if self.with_seg:
            update_losses(self.auxiliary_head_seg, x, gt_semantic_seg[-1], 'aux_seg')

        if self.with_aux_cdt:
            update_losses(self.auxiliary_head_cdt, x, gt_semantic_cdt[-1], 'aux_cdt')

        if self.with_aux_con:
            update_losses(self.auxiliary_head_con, x, gt_semantic_con, 'aux_con')

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        raise NotImplementedError

    @staticmethod
    def process_gt_semantic_con(*gt_semantic_con_list):
        gt_semantic_con = [torch.cat([file.squeeze(1) for file in gt], 1)
                           for gt in gt_semantic_con_list if gt is not None]
        return gt_semantic_con

    def forward_train(self, img, img_metas, gt_semantic_seg=None, gt_semantic_cdt=None,
                      gt_semantic_con_8_d1=None, gt_semantic_con_8_d3=None, gt_semantic_con_8_d5=None):
        """Forward function for training."""
        gt_semantic_con = self.process_gt_semantic_con(
            gt_semantic_con_8_d1, gt_semantic_con_8_d3, gt_semantic_con_8_d5)

        x = self.extract_feat(img)

        losses = dict()
        loss_decode = self._decode_head_forward_train(
            x, img_metas, gt_semantic_seg, gt_semantic_cdt, gt_semantic_con)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, gt_semantic_cdt, gt_semantic_con)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        raise NotImplementedError

    @staticmethod
    def whole_inference_remove(output, resize_shape):
        if isinstance(output, list):
            return [tensor[:, :, :resize_shape[0], :resize_shape[1]] for tensor in output]
        else:
            return output[:, :, :resize_shape[0], :resize_shape[1]]

    def whole_inference_resize(self, output, new_size):
        if isinstance(output, list):
            return [resize(tensor, size=new_size, mode='bilinear',
                           align_corners=self.align_corners, warning=False) for tensor in output]
        else:
            return resize(output, size=new_size, mode='bilinear',
                          align_corners=self.align_corners, warning=False)

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_cdt_con_logit = self.encode_decode(img, img_meta)

        if rescale:
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                # remove padding area
                resize_shape = img_meta[0]['img_shape'][:2]

                seg_cdt_con_logit = [self.whole_inference_remove(sub_out, resize_shape)
                                     if sub_out is not None else None for sub_out in seg_cdt_con_logit]
                size = img_meta[0]['ori_shape'][:2]

            seg_cdt_con_logit = [self.whole_inference_resize(sub_out, size)
                                 if sub_out is not None else None for sub_out in seg_cdt_con_logit]

        return seg_cdt_con_logit

    @staticmethod
    def apply_flip(tensor, flip, flip_direction):
        if flip:
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                tensor = tensor.flip(dims=(3,))
            elif flip_direction == 'vertical':
                tensor = tensor.flip(dims=(2,))
        return tensor

    @staticmethod
    def get_output(logit, out_channels):
        if out_channels == 1:
            output = torch.sigmoid(logit)
        elif out_channels == 9:
            output = torch.sum(logit, dim=1)
        else:
            output = torch.softmax(logit, dim=1)
        return output

    def inference_process(self, logit, out_channels, flip, flip_direction):
        if isinstance(logit, list):
            output = [
                self.apply_flip(self.get_output(sub_logit, out_channels), flip, flip_direction)
                for sub_logit in logit
            ]
        else:
            output = self.apply_flip(self.get_output(logit, out_channels), flip, flip_direction)
        return output

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit, cdt_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit, cdt_logit = self.whole_inference(img, img_meta, rescale)

        flip = img_meta[0]['flip']
        flip_direction = img_meta[0]['flip_direction']

        seg_output = self.inference_process(
            seg_logit, self.out_channels_seg, flip, flip_direction) if self.with_seg else None
        cdt_output = self.inference_process(
            cdt_logit, self.out_channels_cdt, flip, flip_direction) if self.with_cdt else None

        return seg_output, cdt_output

    @staticmethod
    def process_logit(logit, threshold, out_channels):
        if out_channels == 1 or out_channels == 9:
            pred = (logit > threshold).to(logit).squeeze(1)
        else:
            pred = logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only supports 4D output
            return pred.unsqueeze(0)
        pred_out = list(pred.cpu().numpy())[0]
        return pred_out

    def simple_test_process(self, logit, threshold, out_channels):
        if isinstance(logit, list):
            return [self.process_logit(sub_logit, threshold, out_channels)
                    if sub_logit is not None else None for sub_logit in logit]
        return self.process_logit(logit, threshold, out_channels) if logit is not None else None

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""

        seg_logit, cdt_logit = self.inference(img, img_meta, rescale)

        seg_pred = self.simple_test_process(
            seg_logit, self.decode_head_seg.threshold, self.out_channels_seg) if self.with_seg else None

        # fusion
        out_pred = None
        if self.with_ensemble[0]:
            pass
        else:
            out_pred = [seg_pred]

        return out_pred

    def simple_test_logits(self, img, img_metas, rescale=True):
        """Test without augmentations.

        Return numpy seg_map logits.
        """
        raise NotImplementedError

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.
        Only rescale=True is supported.
        """
        raise NotImplementedError

    def aug_test_logits(self, img, img_metas, rescale=True):
        """Test with augmentations.
        Return seg_map logits. Only rescale=True is supported.
        """
        raise NotImplementedError
