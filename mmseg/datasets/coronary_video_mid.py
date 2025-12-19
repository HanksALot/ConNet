import os
import warnings
import numpy as np
from PIL import Image

import mmcv
from mmcv.utils import print_log

from .builder import DATASETS
from .base_dataset import BaseDataset
from .pipelines.video import LoadAnnotationsVideo
from mmseg.core import intersect_and_union
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class CoronaryVideoMidDataset(BaseDataset):
    CLASSES = ('background', 'vessel')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, frame_num=2, gt_seg_map_loader_cfg=None, **kwargs):
        self.frame_num = frame_num
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)

        self.gt_seg_map_loader = LoadAnnotationsVideo(
        ) if gt_seg_map_loader_cfg is None else LoadAnnotationsVideo(
            **gt_seg_map_loader_cfg)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        img_infos = []
        if split is not None:
            raise NotImplemented
        elif self.frame_num != 1:
            img_dir = img_dir.replace('seq_dir', 'img_dir')
            for mid_img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                mid_img_id = int(os.path.basename(os.path.splitext(mid_img)[0]))

                if self.frame_num % 2 == 0:
                    start_value = mid_img_id - self.frame_num // 2
                else:
                    start_value = mid_img_id - (self.frame_num - 1) // 2
                img_id_list = list(range(start_value, start_value + self.frame_num))
                img_id_list.sort()

                img_list = []
                for img_id in img_id_list:
                    img = os.path.join(os.path.split(mid_img)[0], str(img_id)) + self.img_suffix
                    img_list.append(img)
                img_list.sort()
                img_info = dict(filename=img_list)

                if ann_dir is not None:
                    seg_map = [mid_img.replace(img_suffix, seg_map_suffix)]
                    img_info['ann'] = dict(seg_map=seg_map)
                    img_info['ctl'] = dict(ctl_map=seg_map)
                img_infos.append(img_info)
        else:
            raise NotImplemented

        img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def results2img(self, results, imgfile_prefix, indices=None):
        if indices is None:
            indices = list(range(len(self)))

        result_files = []
        for result, idx in zip(results, indices):
            filename = self.img_infos[idx]['filename'][0]
            png_filename = os.path.join(imgfile_prefix, filename)
            mmcv.mkdir_or_exist(os.path.dirname(png_filename))

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       indices=None,
                       to_label_id=False):
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, indices)

        return result_files

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            ann_info = dict(seg_map=[ann_info['seg_map'][-1]])
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results['gt_semantic_seg'][-1]

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)[-1]
            pre_eval_results.append(
                intersect_and_union(
                    pred,
                    seg_map,
                    len(self.CLASSES),
                    self.ignore_index,
                    # as the labels has been converted when dataset initialized
                    # in `get_palette_for_custom_classes ` this `label_map`
                    # should be `dict()`, see
                    # https://github.com/open-mmlab/mmsegmentation/issues/1415
                    # for more ditails
                    label_map=dict(),
                    reduce_zero_label=self.reduce_zero_label))

        return pre_eval_results
