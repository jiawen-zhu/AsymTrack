"""
Backbone modules.
"""
from lib.models.AsymTrack.EfficientMod import efficientMod_xxs
import torch
from torch import nn
from lib.models.AsymTrack import EfficientMod as efficientMod_module


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, open_layers: list, num_channels: int,
                 return_interm_layers: bool, net_type="resnet"):
        super().__init__()
        open_blocks = open_layers[2:]
        open_items = open_layers[0:2]
        for name, parameter in backbone.named_parameters():
            if not train_backbone:
                freeze = True
                for open_block in open_blocks:
                    if open_block in name:
                        freeze = False
                if name in open_items:
                    freeze = False
                if freeze == True:
                    parameter.requires_grad_(False)  # here should allow users to specify which layers to freeze !
        self.body = backbone
        self.num_channels = num_channels

    def forward(self, images_list, train):
        xs = self.body(images_list)
        return xs


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 pretrain_type: str,
                 search_size: int,
                 search_number: int,
                 template_size: int,
                 template_number: int,
                 freeze_bn: bool,
                 neck_type: str,
                 open_layers: list,
                 ckpt_path=None,
                 pretrain=None,
                 output_layers: list = None):
        if 'efficientmod' in name.lower():
            backbone = getattr(efficientMod_module, name)(
                pretrained=pretrain,
                output_layers=output_layers
            )
            num_channels = backbone.num_channels
            net_type = "efficientmod"
        else:
            raise ValueError()
        super().__init__(backbone, train_backbone, open_layers, num_channels, return_interm_layers, net_type=net_type)


def build_backbone(cfg):

    if cfg.MODEL.BACKBONE.TYPE in ['efficientMod_xxs', 'efficientMod_xs', 'efficientMod_s', 'efficientMod_xxs_matt']:
        pretrain = not getattr(cfg, "TEST_MODE", False)
        interact = getattr(cfg.MODEL.BACKBONE, "INTERACT", None)
        extra_module = getattr(cfg.MODEL.BACKBONE, "EXTRA_MODULE", None)

        if cfg.MODEL.BACKBONE.TYPE in ["efficientMod_xxs"]:
            backbone = efficientMod_xxs(pretrained=pretrain, output_layers=cfg.MODEL.BACKBONE.OUTPUT_LAYERS,
                                        interact=interact, extra_module=extra_module, stride=cfg.MODEL.BACKBONE.STRIDE)
        elif cfg.MODEL.BACKBONE.TYPE in ["efficientMod_xxs_matt"]:
            backbone = efficientMod_xxs_matt(pretrained=pretrain, output_layers=cfg.MODEL.BACKBONE.OUTPUT_LAYERS,interact=interact)
        elif cfg.MODEL.BACKBONE.TYPE in ["efficientMod_xs"]:
            backbone = efficientMod_xs(pretrained=pretrain, output_layers=cfg.MODEL.BACKBONE.OUTPUT_LAYERS,interact=interact)
        elif cfg.MODEL.BACKBONE.TYPE in ["efficientMod_s"]:
            backbone = efficientMod_s(pretrained=pretrain, output_layers=cfg.MODEL.BACKBONE.OUTPUT_LAYERS,interact=interact)
    elif cfg.MODEL.BACKBONE.TYPE in ['efficientMod_xxs_separate']:

        pretrain = not getattr(cfg, "TEST_MODE", False)
        interact = getattr(cfg.MODEL.BACKBONE, "INTERACT", None)
        extra_module = getattr(cfg.MODEL.BACKBONE, "EXTRA_MODULE", None)

        backbone = efficientMod_xxs_separate(pretrained=pretrain, output_layers=cfg.MODEL.BACKBONE.OUTPUT_LAYERS,
                                    interact=interact, extra_module=extra_module)

    model = backbone
    model.num_channels = backbone.num_channels

    return model
