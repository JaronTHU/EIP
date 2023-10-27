""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import Tensor
import torch.nn as nn
from typing import List, Union

from .helpers import to_2tuple
from . import create_norm, create_act
from ..build import MODELS


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features, hidden_features=None, out_features=None,
                 act_args={'act': "gelu"}, norm_args=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = create_act(act_args)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_args={'act': "sigmoid"}, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = create_act(act_args)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_args={'act': "gelu"},
                 gate_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = create_act(act_args)
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None,
            act_args={'act': "gelu"},
            norm_args=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = create_norm(norm_args, hidden_features) or nn.Identity()
        self.act = create_act(act_args)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


@MODELS.register_module()
class MLP(nn.Module):
    """General MLP for all purposes

    Warning 1! activations like softmax that activates on a series of values are not supported, since all sizes of inputs are first transformed into (N', C)

    Warning 2! norm can only be batchnorm1d or None

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self, 
        in_features: int,
        hidden_features: List[int],
        out_features: int, 
        act_args: Union[str, dict, List] = {'act': 'relu'},
        norm_args: Union[str, List] = 'bn1d',
        drop: Union[float, List[float]] = 0.,
        order: str = 'norm-act-drop',
        **kwargs
        ) -> None:
        super().__init__()

        num_layers = len(hidden_features) + 1

        if isinstance(act_args, str) or act_args is None:
            act_args = [{'act': act_args}] * (num_layers - 1)
        if isinstance(act_args, dict):
            act_args = [act_args] * (num_layers - 1)
        elif isinstance(act_args, list):
            assert len(act_args) == num_layers - 1, 'Number of activations ({}) plus 1 must match number of layers ({})!'.format(len(act_args), num_layers)
            for i in range(num_layers - 1):
                if isinstance(act_args[i], str) or act_args[i] is None:
                    act_args[i] = {'act': act_args[i]}
        else:
            raise TypeError('Only string, dict and list are supported for "act_args", current {}'.format(type(act_args)))
        
        if isinstance(norm_args, str):
            norm_args = [norm_args] * (num_layers - 1)
        elif isinstance(norm_args, list):
            assert len(norm_args) == num_layers - 1, 'Number of normalizations ({}) plus 1 must match number of layers ({})!'.format(len(norm_args), num_layers)
        else:
            raise TypeError('Only string and list are supported for "norm_args", current {}'.format(type(norm_args)))

        if isinstance(drop, float):
            assert 0. <= drop < 1., 'Dropout must lie in [0, 1), current {}'.format(drop)
            drop = [drop] * (num_layers - 1)
        elif isinstance(drop, list):
            assert min(drop) >= 0. and max(drop) < 1., 'Dropout must lie in [0, 1), current min {}, current max {}'.format(min(drop), max(drop))
            assert len(drop) == num_layers - 1, 'Number of dropouts ({}) plus 1 must match number of layers ({})!'.format(len(drop), num_layers)
        else:
            raise TypeError('Only float and list[float] are supported for "dropout", current {}'.format(type(drop)))

        order = order.split('-')
        order_ = [x for x in order if x != 'norm' and x != 'act' and x != 'drop']
        assert len(order_) == 0, 'Not supported order keywords: {}'.format(order_)
        
        hidden_features = [in_features] + hidden_features
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_features[i], hidden_features[i + 1]))
            if drop[i] == 0. or 'drop' not in order:
                act_args[i]['inplace'] = True
                drop_layer = None
            else:
                act_args[i]['inplace'] = False
                drop_layer = nn.Dropout(drop[i], inplace=True)
            norm_layer = create_norm(norm_args[i], hidden_features[i + 1])
            act_layer = create_act(act_args[i])
            for j in order:
                eval('layers.append({}_layer)'.format(j))
        layers.append(nn.Linear(hidden_features[-1], out_features))
        layers = [layer for layer in layers if layer is not None]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (*, C1)
        :return: (*, C2)
        """
        x_shape = list(x.size())
        x = self.mlp(x.view(-1, x_shape[-1]))
        x_shape[-1] = x.shape[-1]
        x = x.view(*x_shape)
        return x
