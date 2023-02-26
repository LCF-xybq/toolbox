import torch
import torch.nn as nn

from cv.cnn import ConvModule
from cv.utils import get_logger
from cv.runner import load_checkpoint
from ..builder import MODELS


@MODELS.register_module()
class TestNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 kernel_size=3,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN')):
        super(TestNet, self).__init__()
        self.basic_block = ConvModule(
            in_channels=in_channels,
            out_channels=36,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            act_cfg=act_cfg,
            norm_cfg=dict(type='GN', num_groups=3)
        )
        self.conv_out = ConvModule(
            in_channels=36,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            act_cfg=act_cfg,
            norm_cfg=None
        )

    def forward(self, x):
        out = self.basic_block(x)
        out = self.conv_out(out)

        return out

    def init_weights(self, pretrained=None, strict=True):
        if isinstance(pretrained, str):
            logger = get_logger(name='TestNet_init_weights')
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
