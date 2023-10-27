#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2022/11/01 20:53:48
@email: fjjth98@163.com
@description: 
================================================
"""

import torch

from torch.nn.functional import softmax, cross_entropy
from torch.optim.swa_utils import AveragedModel
from ..classification import BaseCls
from ..segmentation import BasePartSeg
from ..build import MODELS, build_model_from_cfg


def gather(input: torch.Tensor, dim: int, index: torch.LongTensor) -> torch.Tensor:
    """A different gather function (torch.gather)

    Args:
        input (torch.Tensor): the source tensor (b1, ..., bn, i1, ..., im)
        dim (int): the axis along which to index (k)
        index (torch.LongTensor): (b1, ..., bn) value [0, ..., ik-1]

    Returns:
        torch.Tensor: (b1, ..., bn, i1, ..., i_{k-1}, i_{k+1}, ..., im)
    """
    output_size = list(input.size())
    output_size[dim] = 1
    index_size = list(index.size()) + [1] * (len(output_size) - len(index.size()))
    return torch.gather(input, dim, index.view(*index_size).expand(*output_size)).squeeze(dim)


@MODELS.register_module()
class BasePoseCls(BaseCls):

    def __init__(self, pose_args=None, encoder_args=None, cls_args=None, criterion_args=None, load_path=None, **kwargs):
        super().__init__(encoder_args, cls_args, criterion_args, **kwargs)
        if load_path is not None:
            # must have encoder, prediction
            self.load_state_dict(torch.load(load_path, map_location='cpu')['model'])
            for p in self.parameters():
                p.requires_grad = False
        self.pose_fn = build_model_from_cfg(pose_args)
        self.eval()
        if load_path is not None:
            self.train = self.pose_fn.train
            self.eval = self.pose_fn.eval

    def forward(self, data):
        b, n, _ = data['pos'].size()
        xyz = self.pose_fn(data['pos'])     # (B, N, 3) or (B, k, N, 3)
        # k = 1 if len(xyz.size()) == 3 else xyz.size(1)
        xyz = xyz.view(-1, n, 3)
        features = xyz.transpose(-2, -1).contiguous()      # (kB, 3, N) 
        global_feat = self.encoder.forward_cls_feat(xyz, features)
        logits = self.prediction(global_feat)   # (kB, C)
        return logits
    
    def get_logits_loss(self, data, gt):
        b, n, _ = data['pos'].size()
        xyz = self.pose_fn(data['pos'])     # (B, N, 3) or (B, k, N, 3)
        k = 1 if len(xyz.size()) == 3 else xyz.size(1)
        xyz = xyz.view(-1, n, 3)
        features = xyz.transpose(-2, -1).contiguous()      # (kB, 3, N) 
        global_feat = self.encoder.forward_cls_feat(xyz, features)
        logits = self.prediction(global_feat)
        gt = gt.long() if len(xyz.size()) == 3 else gt.view(b, 1).repeat(1, k).view(-1).long()
        return logits, self.criterion(logits, gt)

    def forward_pc_feat(self, data):
        b, n, _ = data['pos'].size()
        xyz = self.pose_fn(data['pos'])     # (B, N, 3) or (B, k, N, 3)
        # k = 1 if len(xyz.size()) == 3 else xyz.size(1)
        xyz = xyz.view(-1, n, 3)
        features = xyz.transpose(-2, -1).contiguous()      # (kB, 3, N) 
        global_feat = self.encoder.forward_cls_feat(xyz, features)
        return xyz, global_feat


@MODELS.register_module()
class BasePoseClsNormal(BaseCls):

    def __init__(self, pose_args=None, encoder_args=None, cls_args=None, criterion_args=None, **kwargs):
        super().__init__(encoder_args, cls_args, criterion_args, **kwargs)
        self.pose_fn = build_model_from_cfg(pose_args)

    def forward(self, data):
        r = self.pose_fn.forward_pose(data['pos'])  # (B, 3, 3)
        xyz = data['pos'] @ r # (B, N, 3) or (B, k, N, 3)
        features = torch.cat([xyz.transpose(-2, -1), (data['normals'] @ r).transpose(-2, -1)], dim=1)
        global_feat = self.encoder.forward_cls_feat(xyz, features)
        return self.prediction(global_feat)


@MODELS.register_module()
class BasePosePartSeg(BasePartSeg):
    def __init__(self, pose_args=None, use_normal=False, encoder_args=None, decoder_args=None, cls_args=None, **kwargs):
        super().__init__(encoder_args, decoder_args, cls_args, **kwargs)
        self.use_normal = use_normal
        self.pose_fn = build_model_from_cfg(pose_args)

    def forward(self, p0, f0=None, cls0=None):
        if hasattr(p0, 'keys'):
            p0, f0, cls0 = p0['pos'], p0['x'], p0['cls']
        # (B, N, 3), (B, C, N), (B), C = 3 or 6

        r = self.pose_fn.forward_pose(p0, cls0.squeeze(-1))     # (B, 3, 3) or (B, k, 3, 3)
        if len(r.size()) == 3:
            p0 = p0 @ r
            if self.use_normal:
                n0 = r.transpose(-1, -2) @ f0[:, 3:6]
                f0 = torch.cat([p0.transpose(1, 2), n0], dim=1)     # (B, 6, N)
            else:
                f0 = p0.transpose(1, 2).contiguous()
        else:
            p0 = (p0.unsqueeze(1) @ r).flatten(0, 1)    # (Bk, N, 3)
            if self.use_normal:
                n0 = (r.transpose(-1, -2) @ f0[:, 3:6].unsqueeze(1)).flatten(0, 1)  # (Bk, 3, N)
                f0 = torch.cat([p0.transpose(1, 2), n0], dim=1)     # (Bk, 6, N)
            else:
                f0 = p0.transpose(1, 2).contiguous()

        p = {
            'pos': p0,
            'x': f0,
            'cls': cls0
        }

        p, f = self.encoder.forward_seg_feat(p)
        if self.decoder is not None:
            f = self.decoder(p, f, cls0).squeeze(-1)
        elif isinstance(f, list):
            f = f[-1]
        if self.head is not None:
            f = self.head(f)
        return f
    
    def forward_pose(self, p0, f0=None, cls0=None):
        if hasattr(p0, 'keys'):
            p0, f0, cls0 = p0['pos'], p0['x'], p0['cls']
        # (B, N, 3), (B, C, N), (B), C = 3 or 6

        r = self.pose_fn.forward_pose(p0, cls0.squeeze(-1))     # (B, 3, 3) or (B, k, 3, 3)
        return p0 @ r
