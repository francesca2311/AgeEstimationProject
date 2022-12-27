import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
from SE.se_module import SELayer
from FiLM.film_module import FiLMLayer
from typing import Type, List, Union
import torch

# ResNet layers
# 101: [3, 4, 23, 3]
# 50: [3, 4, 6, 3] ?
# 34: [3, 4, 6, 3] ?
# 18: [2, 2, 2, 2]

def get_SEBottleneck_FiLM(dim_knowledge):
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

def se_resnet18_filmed(dim_knowledge=3) -> nn.Module:
    model = ResNet(get_SEBottleneck_FiLM(dim_knowledge), [2, 2, 2, 2], num_classes=1)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

class SEResNetFiLMedGroups(nn.Module):
    def __init__(self, n_classes: int, dim_knowledge: int) -> None:
        super().__init__()

        # Create the backbone
        self.backbone = se_resnet18_filmed(dim_knowledge)
        # Load into gpu
        self.backbone = self.backbone.to("cuda")
        # Get film layers of the backbone
        self.backbone_film_layers: List[nn.Module] = self._get_film_layers()

        # Replace output FC of backbone
        self.fc0 = nn.Linear(512, n_classes).to("cuda")
        self.backbone.fc = self.fc0

        # Freeze all layers except film layers
        self.set_grads()

    def _get_film_layers(self) -> List:
        backbone_layers = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
        backbone_film_layers = []
        for layer in backbone_layers:
            for layer_module in layer.modules():
                if not layer_module._get_name() == "SEBottleneck_FiLM":
                    continue
                backbone_film_layers += [x for x in layer_module.modules() if x._get_name() == "FiLMLayer"]
        return backbone_film_layers

    def set_grads(self) -> None:
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        for x in self.backbone_film_layers:
            x.requires_grad_(True)
        self.fc0.requires_grad_(True)

    def set_knowledge(self, knowledge: torch.Tensor) -> None:
        for x in self.backbone_film_layers:
            x.set_knowledge(knowledge)

    def forward(self, x, knowledge):
        self.set_knowledge(knowledge)
        out = self.backbone(x)
        return out