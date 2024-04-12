from typing import Dict, List, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from util.misc import NestedTensor, is_main_process, get_cuda_memory_usage, mem_info_logger

from .position_encoding import build_position_encoding


from types import FunctionType


def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed

    Args:
        num_features (int): Number of features ``C`` from an expected input of size ``(N, C, H, W)``
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: Tensor) -> Tensor:
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        logging.debug("{} {}".format(get_cuda_memory_usage(), "After FrozenBatchNorm2d weight."))

        b = self.bias.reshape(1, -1, 1, 1)
        logging.debug("{} {}".format(get_cuda_memory_usage(), "After FrozenBatchNorm2d bias."))

        rv = self.running_var.reshape(1, -1, 1, 1)
        logging.debug("{} {}".format(get_cuda_memory_usage(), "After FrozenBatchNorm2d rv."))

        rm = self.running_mean.reshape(1, -1, 1, 1)
        logging.debug("{} {}".format(get_cuda_memory_usage(), "After FrozenBatchNorm2d rm."))
        
        scale = w * (rv + self.eps).rsqrt()
        logging.debug("{} {}".format(get_cuda_memory_usage(), "After FrozenBatchNorm2d scale."))
        
        bias = b - rm * scale
        logging.debug("{} {}".format(get_cuda_memory_usage(), "After FrozenBatchNorm2d bias."))
        return x * scale + bias

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps})"



class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool, args):
        super().__init__()
        self.args = args
        self.backbone= backbone

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

        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    @mem_info_logger('Support Encoding Net of Backbone')
    def support_encoding_net(self, x, return_interm_layers=False):
        # TODO Codes below is not so good.
        out: Dict[str, NestedTensor] = {}
        self.backbone: torchvision.models.ResNet
        m = x.mask
        # x = self.meta_conv(x.tensors)
        x = self.backbone.conv1(x.tensors)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        
        x = self.backbone.layer2(x)
        if return_interm_layers:
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out['0'] = NestedTensor(x, mask)

        x = self.backbone.layer3(x) # TODO !!! Out of memory here.

        if return_interm_layers:
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out['1'] = NestedTensor(x, mask)

        x = self.backbone.layer4(x)
        if return_interm_layers:
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out['2'] = NestedTensor(x, mask)

        if return_interm_layers:
            return out
        else:
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out['0'] = NestedTensor(x, mask)
            return out

    @mem_info_logger('Backbone')
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
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
        # norm_layer = FrozenBatchNorm2d
        norm_layer = torch.nn.BatchNorm2d
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
    def forward_supp_branch(self, tensor_list: NestedTensor, return_interm_layers=False):
        # ! The next line gets the features produced by the feature extractor(resnet here).
        # ! return_interm_layers determine how many layers to use and how many out features produced.
        logging.debug("{} {}".format(get_cuda_memory_usage(), "Before backbone support encoding net."))
        xs = self[0].support_encoding_net(tensor_list, return_interm_layers=return_interm_layers)
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
    return_interm_layers = (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args)
    model = Joiner(backbone, position_embedding)
    return model
