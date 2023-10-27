#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2022/10/30 21:15:59
@email: fjjth98@163.com
@description: 
================================================
"""

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.linalg import vector_norm
from torch.nn.functional import normalize, relu

from ..build import MODELS, build_model_from_cfg
from ..layers import grouping_operation


@MODELS.register_module()
class EIP(Module):
    """Transformation-Invariant Poses for Point Cloud Processing (K poses)

    Args:
        Module (_type_):
    """

    def __init__(self, regress_args) -> None:
        super().__init__()
        self.reg = build_model_from_cfg(regress_args)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): coordinates (*, N, 3)
            y (torch.Tensor, optional): other features besides coordinates (*, N, C1). Defaults to None.

        Returns:
            torch.Tensor: rotation matrix (*, k, N, 3) or (*, N, 3)
        """

        r = self.forward_pose(x, y)
        if len(r.size()) == len(x.size()) + 1:
            x = x.unsqueeze(-3)
        return x @ r

    def forward_pose(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): (*, N, 3)
            y (torch.Tensor, optional): (*, N, C). Defaults to None.

        Returns:
            torch.Tensor: (*, k, 3, 3) or (*, 3, 3)
        """
        x = x - x.mean(dim=-2, keepdim=True)    # (*, N, 3)
        inv_feat = self.inv_feat(x)
        # if y is not None:
        #     z = self.reg(inv_feat, y)
        z = self.reg(inv_feat)
        r = self.orthonormalize(z.transpose(-2, -1) @ x)      # (*, k, 3, 3)
        return r.squeeze(-3)

    def inv_feat(self, x: torch.Tensor) -> torch.Tensor:
        """Generate invariant features for poses

        Args:
            x (torch.Tensor): coordinates (*, N, 3)

        Returns:
            torch.Tensor: (*, N, C2). In this implementation C2=1
        """
        return vector_norm(x, dim=-1, keepdim=True)

    def orthonormalize(self, xz: torch.Tensor) -> torch.Tensor:
        """Orthonormalize

        Args:
            xz (torch.Tensor): (*, 2*k, 3)

        Returns:
            torch.Tensor: rotation matrix (*, k, 3, 3)
        """
        xz = xz.view(*xz.size()[:-2], -1, 2, 3)     # (*, k, 2, 3)
        r1 = normalize(xz[..., 0, :], dim=-1)   # (*, k, 3)
        r2 = normalize(torch.cross(xz[..., 0, :], xz[..., 1, :], dim=-1), dim=-1)       # (*, k, 3)
        r3 = torch.cross(r1, r2, dim=-1)    # (*, k, 3)
        return torch.stack([r1, r2, r3], dim=-1)    # (*, k, 3, 3)


@MODELS.register_module()
class EIPTable(Module):
    """Parameter able for [0,1]->R^n

    Args:
        Module (_type_): _description_
    """
    def __init__(self, bins: int, out_features: int) -> None:
        super().__init__()
        self.bins = bins
        self.table = Parameter(torch.randn(bins, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): (*, N, 1)

        Returns:
            torch.Tensor: (*, N, C)
        """
        idx = (x * self.bins).floor().long().squeeze(-1)
        idx[idx >= self.bins] = self.bins - 1
        return self.table[idx]
