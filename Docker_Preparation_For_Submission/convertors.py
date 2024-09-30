import copy
import timm
import torch
import types
import torch.nn as nn

from collections import OrderedDict
from nnspt.segmentation.unetpp import Unetpp

def __classinit(cls):
    return cls._init__class()

def __is_generator_empty(generator):
    try:
        next(generator)
        return False
    except StopIteration:
        return True

def convert_inplace(net, convertor):
    stack = [net]

    while stack:
        node = stack[-1]

        stack.pop()

        for name, child in node.named_children():
            if not __is_generator_empty(child.children()):
                stack.append(child)

            setattr(node, name, convertor(child))

@__classinit
class LayerConvertor(object):
    @classmethod
    def _init__class(cls):
        cls._registry = { }

        return cls()

    def __call__(self, layer):
        if type(layer) in self._registry:
            return self._registry[type(layer)](layer)
        else:
            return self._func_None(layer)

    @classmethod
    def _func_None(cls, layer):
        return layer

@__classinit
class LayerConvertorNNSPT(LayerConvertor.__class__):
    @classmethod
    def _init__class(cls):
        cls._registry = {
            nn.Conv1d: getattr(cls, '_func_Conv1d'),
            nn.MaxPool1d: getattr(cls, '_func_MaxPool1d'),
            nn.AvgPool1d: getattr(cls, '_func_AvgPool1d'),
            nn.BatchNorm1d: getattr(cls, '_func_BatchNorm1d'),
            nn.AdaptiveAvgPool1d: getattr(cls, '_func_AdaptiveAvgPool1d')
        }

        return cls()

    @staticmethod
    def __expand_tuple(param):
        return (param[0], param[0], param[0])

    @classmethod
    def _func_None(cls, layer):
        return layer

    @classmethod
    def _func_AdaptiveAvgPool1d(cls, layer1d):
        kwargs = {
            'output_size': layer1d.output_size
        }

        layer3d = nn.AdaptiveAvgPool3d(**kwargs)
        return layer3d

    @classmethod
    def _func_Conv1d(cls, layer1d):
        kwargs = {
            'in_channels': layer1d.in_channels,
            'out_channels': layer1d.out_channels,
            'kernel_size': cls.__expand_tuple(layer1d.kernel_size),
            'stride': cls.__expand_tuple(layer1d.stride),
            'padding': cls.__expand_tuple(layer1d.padding),
            'dilation': cls.__expand_tuple(layer1d.dilation),
            'groups': layer1d.groups,
            'bias': 'bias' in layer1d.state_dict(),
            'padding_mode': layer1d.padding_mode
        }

        layer3d = nn.Conv3d(**kwargs)

        return layer3d

    @classmethod
    def _func_BatchNorm1d(cls, layer1d):
        kwargs = {
            'num_features': layer1d.num_features,
            'eps': layer1d.eps,
            'momentum': layer1d.momentum,
            'affine': layer1d.affine,
            'track_running_stats': layer1d.track_running_stats
        }

        layer3d = nn.BatchNorm3d(**kwargs)

        return layer3d

    @classmethod
    def _func_MaxPool1d(cls, layer1d):
        kwargs = {
            'kernel_size': layer1d.kernel_size,
            'stride': layer1d.stride,
            'padding': layer1d.padding,
            'dilation': layer1d.dilation,
            'return_indices': layer1d.return_indices,
            'ceil_mode': layer1d.ceil_mode
        }

        layer3d = nn.MaxPool3d(**kwargs)

        return layer3d

    @classmethod
    def _func_AvgPool1d(cls, layer1d):
        kwargs = {
            'kernel_size': layer1d.kernel_size,
            'stride': layer1d.stride,
            'padding': layer1d.padding,
            'ceil_mode': layer1d.ceil_mode,
            'count_include_pad': layer1d.count_include_pad,
            'divisor_override': layer1d.divisor_override
        }

        layer3d = nn.AvgPool3d(**kwargs)

        return layer3d

@__classinit
class LayerConvertorSm(LayerConvertor.__class__):
    @classmethod
    def _init__class(cls):
        cls._registry = {
            timm.layers.grn.GlobalResponseNorm: getattr(cls, '_func_timm_GlobalResponseNorm'),
            timm.models._efficientnet_blocks.SqueezeExcite: getattr(cls, '_func_timm_SqueezeExcite'),
            timm.layers.norm_act.BatchNormAct2d: getattr(cls, '_func_timm_layers_norm_act_BatchNormAct2d'),
            timm.layers.norm.LayerNorm2d: getattr(cls, '_func_timm_layers_norm_LayerNorm2d'),
        }

        return cls()

    @classmethod
    def _func_timm_GlobalResponseNorm(cls, layer):
        if layer.channel_dim == -1:
            layer.spatial_dim = (1, 2, 3)
            layer.wb_shape = (1, 1, 1, 1, -1)
        else:
            layer.spatial_dim = (2, 3, 4)
            layer.wb_shape = (1, -1, 1, 1, 1)

        return layer

    @staticmethod
    def _timm_squeezeexcite_forward(self, x):
        """
            :NOTE:
                it is a copy of timm.layers.squeeze_excite.SEModule function with correct operations under dims
        """
        x_se = x.mean((2, 3, 4), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

    @classmethod
    def _func_timm_SqueezeExcite(cls, layer):
        layer.forward = types.MethodType(cls._timm_squeezeexcite_forward, layer)

        return layer

    @staticmethod
    def _timm_layers_norm_act_batchnormact2d_forward(self, x):
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = nn.functional.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

        x = self.drop(x)
        x = self.act(x)

        return x

    @classmethod
    def _func_timm_layers_norm_act_BatchNormAct2d(cls, layer):
        layer.forward = types.MethodType(cls._timm_layers_norm_act_batchnormact2d_forward, layer)

        return layer

    @staticmethod
    def _timm_layers_norm_layernorm2d_forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)

        return x

    @classmethod
    def _func_timm_layers_norm_LayerNorm2d(cls, layer):
        layer.forward = types.MethodType(cls._timm_layers_norm_layernorm2d_forward, layer)

        return layer

@__classinit
class EvalFusing(object):
    @classmethod
    def _init__class(cls):
        cls._registry = {
            'timm-efficientnetv2-m':
                [ ('convbn_timm', ('conv_stem', 'bn1')) ] + \
                [ ('convbn_timm', (f'blocks.0.{i}.conv'    , f'blocks.0.{i}.bn1')) for i in range(3)  ] + \
                [ ('convbn_timm', (f'blocks.1.{i}.conv_exp', f'blocks.1.{i}.bn1')) for i in range(5)  ] + \
                [ ('convbn_timm', (f'blocks.1.{i}.conv_pwl', f'blocks.1.{i}.bn2')) for i in range(5)  ] + \
                [ ('convbn_timm', (f'blocks.2.{i}.conv_exp', f'blocks.2.{i}.bn1')) for i in range(5)  ] + \
                [ ('convbn_timm', (f'blocks.2.{i}.conv_pwl', f'blocks.2.{i}.bn2')) for i in range(5)  ] + \
                [ ('convbn_timm', (f'blocks.3.{i}.conv_pw' , f'blocks.3.{i}.bn1')) for i in range(7)  ] + \
                [ ('convbn_timm', (f'blocks.3.{i}.conv_dw' , f'blocks.3.{i}.bn2')) for i in range(7)  ] + \
                [ ('convbn_timm', (f'blocks.3.{i}.conv_pwl', f'blocks.3.{i}.bn3')) for i in range(7)  ] + \
                [ ('convbn_timm', (f'blocks.4.{i}.conv_pw' , f'blocks.4.{i}.bn1')) for i in range(14) ] + \
                [ ('convbn_timm', (f'blocks.4.{i}.conv_dw' , f'blocks.4.{i}.bn2')) for i in range(14) ] + \
                [ ('convbn_timm', (f'blocks.4.{i}.conv_pwl', f'blocks.4.{i}.bn3')) for i in range(14) ] + \
                [ ('convbn_timm', (f'blocks.5.{i}.conv_pw' , f'blocks.5.{i}.bn1')) for i in range(18) ] + \
                [ ('convbn_timm', (f'blocks.5.{i}.conv_dw' , f'blocks.5.{i}.bn2')) for i in range(18) ] + \
                [ ('convbn_timm', (f'blocks.5.{i}.conv_pwl', f'blocks.5.{i}.bn3')) for i in range(18) ] + \
                [ ('convbn_timm', (f'blocks.6.{i}.conv_pw' , f'blocks.6.{i}.bn1')) for i in range(5)  ] + \
                [ ('convbn_timm', (f'blocks.6.{i}.conv_dw' , f'blocks.6.{i}.bn2')) for i in range(5)  ] + \
                [ ('convbn_timm', (f'blocks.6.{i}.conv_pwl', f'blocks.6.{i}.bn3')) for i in range(5)  ]
        }

        cls._fusers = {
            'convbn_timm': getattr(cls, '_fuse_conv_bn_timm'),
        }

        return cls()

    @staticmethod
    def _get_module(model, submodule_key):
        tokens = submodule_key.split('.')
        cur_mod = model
        for s in tokens:
            cur_mod = getattr(cur_mod, s)
        return cur_mod
    
    @staticmethod
    def _set_module(model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)

        setattr(cur_mod, tokens[-1], module)
    
    @staticmethod
    def _fuse_conv_bn_torch(fmodel, mnames):
        conv = EvalFusing._get_module(fmodel, mnames[0])
        bn = EvalFusing._get_module(fmodel, mnames[1])

        weight, bias = torch.nn.utils.fuse_conv_bn_weights(
            conv.weight,
            conv.bias,
            bn.running_mean,
            bn.running_var,
            bn.eps,
            bn.weight,
            bn.bias,
        )
        
        conv.weight = weight
        conv.bias = bias

        bn = torch.nn.Identity()

        EvalFusing._set_module(fmodel, mnames[1], bn)

    @staticmethod
    def _fuse_conv_bn_timm(fmodel, mnames):
        conv = EvalFusing._get_module(fmodel, mnames[0])
        bn = EvalFusing._get_module(fmodel, mnames[1])

        weight, bias = torch.nn.utils.fuse_conv_bn_weights(
            conv.weight,
            conv.bias,
            bn.running_mean,
            bn.running_var,
            bn.eps,
            bn.weight,
            bn.bias,
        )
        
        conv.weight = weight
        conv.bias = bias

        if isinstance(bn.drop, torch.nn.Identity):
            bn = bn.act
        else:
            bn = torch.nn.Sequential(bn.drop , bn.act)

        EvalFusing._set_module(fmodel, mnames[1], bn)
        
    def __call__(cls, model, type_):
        if type_ in cls._registry:
            fmodel = model.eval()

            for fname, mnames in cls._registry[type_]:
                cls._fusers[fname](fmodel, mnames)
                
            return fmodel
        else:
            return model.eval()

def create_rostepifanov_model():
    model = Unetpp( in_channels=1,
                out_channels=24,
                encoder='timm-efficientnetv2-m')

    convert_inplace(model, LayerConvertorNNSPT)
    convert_inplace(model, LayerConvertorSm)

    def decoder_forward(self, *feats):
        xs = dict()

        for idx, x in enumerate(feats):
            xs[f'x_{idx}_{idx-1}'] = x

        for idx in range(self.nblocks):
            for jdx in range(self.nblocks - idx):
                depth = jdx
                layer = idx+jdx

                block = self.blocks[f'b_{depth}_{layer}']

                if depth == 0:
                    skip = None
                    shape = xs[f'x_{0}_{-1}'].shape
                else:
                    skip = torch.concat([ xs[f'x_{depth}_{layer-sdx-1}'] for sdx in range(layer-depth+1) ], axis=1)
                    shape = xs[f'x_{depth}_{layer-1}'].shape

                x = xs[f'x_{depth+1}_{layer}']
                x = block(x, skip, shape)
                xs[f'x_{depth}_{layer}'] = x
                if depth == 0 and layer == self.nblocks - 1:
                    return xs

        return xs

    heads = {}

    for idx in range(model.decoder.nblocks):
        heads[f'{idx}'] = copy.deepcopy(model.head)

    model.heads = torch.nn.ModuleDict(heads)

    del model.head

    def model_forward(self, x):
        f = self.encoder(x)
        xs = self.decoder(*f)
        idx = self.decoder.nblocks - 1
        x = xs[f'x_{0}_{idx}']
        x = self.heads[f'{idx}'](x)
        return x

    import types

    model.decoder.forward = types.MethodType(decoder_forward, model.decoder)
    model.forward = types.MethodType(model_forward, model)

    del model.decoder.blocks.b_0_0.attention1
    del model.decoder.blocks.b_0_1.attention1
    del model.decoder.blocks.b_0_2.attention1
    del model.decoder.blocks.b_0_3.attention1
    del model.decoder.blocks.b_0_4.attention1

    return model


import numpy as np
from itertools import product, chain

def voxel_sequential_selector(voxel_shape, case_keys, shapes, steps):
    assert type(case_keys) in {list, tuple}
    assert len(voxel_shape) == len(steps)

    for idx, case_key in enumerate(case_keys):
        ranges = ()

        for step, voxel_size, case_size in zip(steps, voxel_shape, shapes[idx]):
            assert case_size >= voxel_size

            range_ = chain(np.arange(0, case_size - voxel_size, step), (case_size - voxel_size ,))
            ranges = (*ranges, range_)

        for point in product(*ranges):
            selector = ()

            for coord, voxel_size in zip(point, voxel_shape):
                selector = (*selector, slice(coord, coord+voxel_size))

            yield case_key, selector