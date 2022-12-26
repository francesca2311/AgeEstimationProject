import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
from SE.se_module import SELayer
from FiLM.film_module import FiLMLayer
from typing import Type, List, Union
import torch


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def get_SEBottleneck_FiLM(dim_knowledge=81):
    class SEBottleneck_FiLM(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                    base_width=64, dilation=1, norm_layer=None,
                    *, reduction=16):
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
            self.film1 = FiLMLayer(dim_knowledge, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            self.relu = nn.ReLU(inplace=True)
            self.se = SELayer(planes * 4, reduction)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.film1(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.se(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out
    return SEBottleneck_FiLM

def se_resnet101_filmed(num_classes=1_000, dim_knowledge=81) -> nn.Module:
    """Constructs a ResNet-101 filmed model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(get_SEBottleneck_FiLM(dim_knowledge), [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def se_resnet50_filmed(num_classes=1_000, dim_knowledge=81, pretrained=False) -> nn.Module:
    """Constructs a ResNet-101 filmed model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(get_SEBottleneck_FiLM(dim_knowledge), [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"), strict=False)
    return model

def se_resnet18_filmed(num_classes=1_000, dim_knowledge=81, pretrained=False) -> nn.Module:
    """Constructs a ResNet-101 filmed model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(get_SEBottleneck_FiLM(dim_knowledge), [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

# def se_resnet18(num_classes=1_000):
#     """Constructs a ResNet-18 model.
# 
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
#     model.avgpool = nn.AdaptiveAvgPool2d(1)
#     return model
# 
# 
# def se_resnet34(num_classes=1_000):
#     """Constructs a ResNet-34 model.
# 
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
#     model.avgpool = nn.AdaptiveAvgPool2d(1)
#     return model