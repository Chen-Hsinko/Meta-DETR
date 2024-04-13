from typing import Dict, List, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
from torchvision.ops import FrozenBatchNorm2d
from torchvision.models._utils import IntermediateLayerGetter

from util.misc import NestedTensor, is_main_process, get_cuda_memory_usage, mem_info_logger

from .position_encoding import build_position_encoding
from .efficientdet import BiFPN


from types import FunctionType


class ResNetBiFPN(nn.Module):
    def __init__(self, phi):
        super().__init__()
        #--------------------------------#
        self.phi = phi
        #---------------------------------------------------#
        #   backbone_phi指的是该efficientdet对应的efficient
        #---------------------------------------------------#
        self.backbone_phi = [0, 1, 
                             2, 3, 4, 5, 6, 6
                             ]
        #--------------------------------#
        #   BiFPN所用的通道数
        #--------------------------------#
        self.fpn_num_filters = [256, 384, 
                                512, 640, 768, 768
                                ]
        #--------------------------------#
        #   BiFPN的重复次数
        #--------------------------------#
        self.fpn_cell_repeats = [3, 4, 
                                 5, 6, 7, 7, 8, 8
                                 ]
        #---------------------------------------------------#
        conv_channel_coef = {
            0: [512, 1024, 2048],
            1: [512, 1024, 2048],
            # 2: [48, 120, 352],
            # 3: [48, 136, 384],
            # 4: [56, 160, 448],
            # 5: [64, 176, 512],
            # 6: [72, 200, 576],
            # 7: [72, 200, 576],
        }

        #------------------------------------------------------#
        #   在经过多次BiFPN模块的堆叠后，我们获得的fpn_features
        #   假设我们使用的是efficientdet-D0包括五个有效特征层：
        #   P3_out      64,64,64
        #   P4_out      32,32,64
        #   P5_out      16,16,64
        #   P6_out      8,8,64
        #   P7_out      4,4,64
        #------------------------------------------------------#
        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.phi],
                    conv_channel_coef[phi],
                    True if _ == 0 else False,
                    attention=True if phi < 6 else False)
              for _ in range(self.fpn_cell_repeats[phi])])
    
    def forward(self, features):
        return self.bifpn(features)


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool, args):
        super().__init__()
        self.args = args
        self.backbone= backbone
        self.return_interm_layers = return_interm_layers

        # Settings for freezing backbone
        assert 0 <= args.freeze_backbone_at_layer <= 4
        for name, parameter in backbone.named_parameters(): parameter.requires_grad_(False)  # First freeze all
        if train_backbone:
            if args.freeze_backbone_at_layer == 0:
                for name, parameter in backbone.named_parameters():
                    if 'layer1' in name or 'layer2' in name or 'layer3' in name or 'layer4' in name:
                        parameter.requires_grad_(True)
            elif args.freeze_backbone_at_layer == 1:
                for name, parameter in backbone.named_parameters():
                    if 'layer2' in name or 'layer3' in name or 'layer4' in name:
                        parameter.requires_grad_(True)
            elif args.freeze_backbone_at_layer == 2:
                for name, parameter in backbone.named_parameters():
                    if 'layer3' in name or 'layer4' in name:
                        parameter.requires_grad_(True)
            elif args.freeze_backbone_at_layer == 3:
                for name, parameter in backbone.named_parameters():
                    if 'layer4' in name:
                        parameter.requires_grad_(True)
            elif args.freeze_backbone_at_layer == 4:
                pass
            else:
                raise RuntimeError

        # if return_interm_layers:
        #     return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        #     self.strides = [8, 16, 32]
        #     self.num_channels = [512, 1024, 2048]
        # else:
        #     return_layers = {'layer4': "0"}
        #     self.strides = [32]
        #     self.num_channels = [2048]
        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.body = IntermediateLayerGetter(backbone, return_layers={"layer2": "0", "layer3": "1", "layer4": "2"})

        self.bifpn = ResNetBiFPN(args.phi)

    @mem_info_logger('Support Encoding Net of Backbone')
    def support_encoding_net(self, tensor_list):
        out: Dict[str, NestedTensor] = {}
        xs = self.body(tensor_list.tensors)
        xs = dict(zip(
            ['P3', 'P4', 'P5', 'P6', 'P7'], 
            self.bifpn(list(xs.values()))
        ))
        
        if self.return_interm_layers:
            for name, x in xs.items():
                m = tensor_list.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                out[name] = NestedTensor(x, mask)
            return out
        else:
            mask = F.interpolate(m[None].float(), size=xs['P5'].shape[-2:]).to(torch.bool)[0]
            out['0'] = NestedTensor(xs['P5'], mask)
            return out
            

    @mem_info_logger('Backbone')
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        xs = dict(zip(['P3', 'P4', 'P5', 'P6', 'P7'], self.bifpn(list(xs.values()))))
        out: Dict[str, NestedTensor] = {}
        if self.return_interm_layers:
            for name, x in xs.items():
                m = tensor_list.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                out[name] = NestedTensor(x, mask)
            return out
        else:
            mask = F.interpolate(m[None].float(), size=xs['P5'].shape[-2:]).to(torch.bool)[0]
            out['0'] = NestedTensor(xs['P5'], mask)
            return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 args):
        self.args = args
        dilation = args.dilation
        norm_layer = FrozenBatchNorm2d
        # norm_layer = torch.nn.BatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded, cannot use res18 & res34."
        super().__init__(backbone, train_backbone, return_interm_layers, args)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    @mem_info_logger('Joiner')
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

    @mem_info_logger('Support Branch of Joiner')
    def forward_supp_branch(self, tensor_list: NestedTensor):
        # ! The next line gets the features produced by the feature extractor(resnet here).
        # ! return_interm_layers determine how many layers to use and how many out features produced.
        logging.debug("{} {}".format(get_cuda_memory_usage(), "Before backbone support encoding net."))
        xs = self[0].support_encoding_net(tensor_list)
        logging.debug("{} {}".format(get_cuda_memory_usage(), "After backbone support encoding net."))

        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    # return_interm_layers = (args.num_feature_levels > 1)
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args)
    model = Joiner(backbone, position_embedding)
    return model
