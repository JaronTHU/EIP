#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2023/03/11 16:44:10
@email: fjjth98@163.com
@description: finite rotation group
================================================
"""

import math
import torch

from torch.nn.functional import normalize
from pytorch3d.transforms import axis_angle_to_matrix


def tetrahedral() -> torch.Tensor:
    """Tetrahedral group"""

    res = [torch.eye(3)]

    v = torch.tensor([
        [1., 0., 0.],
        [-0.5, math.sqrt(0.75), 0.],
        [-0.5, -math.sqrt(0.75), 0.],
        [0., 0., math.sqrt(2)]
    ])
    v -= v.mean(dim=0, keepdim=True)

    angle = math.pi * 2 / 3
    for i in range(4):
        axis = normalize(v[i], dim=0)
        res.append(axis_angle_to_matrix(angle * axis))
        res.append(axis_angle_to_matrix(2 * angle * axis))

    angle = math.pi
    for axis in [
        normalize(v[0] + v[1] - v[2] - v[3], dim=0),
        normalize(v[0] - v[1] + v[2] - v[3], dim=0),
        normalize(v[0] - v[1] - v[2] + v[3], dim=0)
    ]:
        res.append(axis_angle_to_matrix(angle * axis))

    return torch.stack(res, dim=0)


def octahedral() -> torch.Tensor:
    """Octahedral group"""

    res = [torch.eye(3)]

    for i in range(1, 4):
        angle = math.pi / 2 * i
        res.append(axis_angle_to_matrix(torch.tensor([angle, 0., 0.])))
        res.append(axis_angle_to_matrix(torch.tensor([0., angle, 0.])))
        res.append(axis_angle_to_matrix(torch.tensor([0., 0., angle])))

    angle = torch.pi
    for i in [-1., 1.]:
        for axis in [
            normalize(torch.tensor([1., i, 0.]), dim=0),
            normalize(torch.tensor([1., 0., i]), dim=0),
            normalize(torch.tensor([0., 1., i]), dim=0)
        ]:
            res.append(axis_angle_to_matrix(angle * axis))

    angle = torch.pi * 2 / 3
    for axis in [
        normalize(torch.tensor([1., 1., 1.]), dim=0),
        normalize(torch.tensor([1., 1., -1.]), dim=0),
        normalize(torch.tensor([1., -1., 1.]), dim=0),
        normalize(torch.tensor([-1., 1., 1.]), dim=0)
    ]:
        res.append(axis_angle_to_matrix(angle * axis))
        res.append(axis_angle_to_matrix(2 * angle * axis))

    return torch.stack(res, dim=0)
        

def icosahedral() -> torch.Tensor:

    res = [torch.eye(3)]
    
    phi = 0.5 * (1 + math.sqrt(5))

    v = torch.tensor([
        [0, 1, phi],
        [0, -1, phi],
        [phi, 0, 1],
        [-phi, 0, 1],
        [1, phi, 0],
        [-1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [phi, 0, -1],
        [-phi, 0, -1],
        [0, 1, -phi],
        [0, -1, -phi]
    ])

    # 20
    angle = torch.pi * 2 / 3
    for t in [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 4],
        [0, 4, 5],
        [0, 3, 5],
        [1, 3, 6],
        [1, 6, 7],
        [1, 2, 7],
        [2, 4, 8],
        [2, 7, 8]
    ]:
        axis = normalize(v[t[0]] + v[t[1]] + v[t[2]], dim=0)
        res.append(axis_angle_to_matrix(angle * axis))
        res.append(axis_angle_to_matrix(2 * angle * axis))

    # 15
    angle = torch.pi
    for l in [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [1, 2],
        [1, 3],
        [1, 6],
        [1, 7],
        [2, 4],
        [2, 7],
        [2, 8],
        [3, 5],
        [3, 6],
        [4, 5]
    ]:
        axis = normalize(v[l[0]] + v[l[1]], dim=0)
        res.append(axis_angle_to_matrix(angle * axis))

    # 24
    angle = torch.pi * 2 / 5
    for i in range(6):
        axis = normalize(v[i], dim=0)
        for j in range(1, 5):
            res.append(axis_angle_to_matrix(j * angle * axis))

    return torch.stack(res, dim=0)


def random_rot(b):
    x = torch.randn(2, b, 3)
    r1 = normalize(x[0], dim=-1)
    r2 = normalize(torch.cross(r1, x[1], dim=-1), dim=-1)
    r3 = torch.cross(r1, r2, dim=-1)
    return torch.stack([r1, r2, r3], dim=-1)

