"""
Basic AsymTrack Model
"""
import pdb

import torch
from torch import nn
from .backbone import build_backbone
from .head import build_box_head
from .neck import build_neck
from lib.utils.box_ops import box_xyxy_to_cxcywh


class AsymTrack(nn.Module):
    """ This is the base class for Transformer Tracking """
    def __init__(self, backbone, box_head, hidden_dim, num_queries, backbone_name,
                 bottleneck=None, aux_loss=False, head_type="CORNER", neck_type='LINEAR'):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name

        self.neck_type = neck_type
        self.box_head = box_head
        self.num_queries = num_queries
        self.bottleneck = bottleneck
        self.aux_loss = aux_loss
        self.head_type = head_type
        if "CORNER" in head_type:
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, images_list=None, xz=None, mode="backbone", run_box_head=True, run_cls_head=False, train=True):
        if mode == "backbone":
            return self.forward_backbone(images_list,train)
        elif mode == "head":
            return self.forward_head(xz, run_box_head=run_box_head, run_cls_head=run_cls_head)
        else:
            raise ValueError

    def forward_backbone(self, images_list, train):
        # Forward the backbone
        xz = self.backbone(images_list, train)  # features & masks, position embedding for the search
        return xz

    def forward_head(self, xz, run_box_head=True, run_cls_head=False):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")

        if self.neck_type == 'LINEAR':
            x = xz[-1]
            B, H, W, C = x.shape
            xz_mem = self.bottleneck(x.view(B, H * W, C))
        elif self.neck_type == 'FPN':
            xz_mem = self.bottleneck(xz)
        elif self.neck_type == 'None':
            xz_mem = xz
        else:
            xz_mem = xz[-1].permute(1, 0, 2)
            xz_mem = self.bottleneck(xz_mem)

        if 'efficientmod' in self.backbone_name:
            output_embed = None  # query token in STARK
            x_mem = xz_mem

        # Forward the corner head
        out, outputs_coord = self.forward_box_head(output_embed, x_mem)
        return out, outputs_coord, output_embed

    def forward_box_head(self, hs, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if "CORNER" in self.head_type:

            enc_opt = memory

            opt = enc_opt.unsqueeze(-1).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            bs, Nq, C, HW = opt.size()

            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))

            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new}

            return out, outputs_coord_new

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]


def build_asymtrack(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    box_head = build_box_head(cfg)

    if 'efficientmod' in cfg.MODEL.BACKBONE.TYPE.lower():
        bottleneck = build_neck(cfg, backbone.num_channels, backbone.num_patches_search, backbone.embed_dim)
    else:
        raise ValueError("illegal backbone type")

    model = AsymTrack(
        backbone,
        box_head,
        bottleneck = bottleneck,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE,
        neck_type=cfg.MODEL.NECK.TYPE,
        backbone_name=cfg.MODEL.BACKBONE.TYPE.lower()
    )

    return model
