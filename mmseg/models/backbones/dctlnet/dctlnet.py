from mmcv.runner import BaseModule
from mmseg.models.builder import BACKBONES

from .linknet3d_iter import DCtlLinkNet3DIter


@BACKBONES.register_module()
class DCtlNet(BaseModule):
    def __init__(self, class_of_model, temporal_kernel, num_iters, init_cfg=None, **kwargs):
        super().__init__(init_cfg)
        assert class_of_model in ['LinkNet', 'LinkNet3DIter', 'LinkNet3D2DStack', 'LinkNet3D2DLoop']

        if class_of_model == 'LinkNet3DIter':
            self.dctlnet = DCtlLinkNet3DIter(temporal_kernel=temporal_kernel, num_iters=num_iters, **kwargs)
        else:
            raise NotImplementedError

    def forward(self, x):
        dec_outs = self.dctlnet(x)
        return dec_outs
