from typing import Optional, Callable, List
import torch
from torch import Tensor
from torch import nn
from torchvision.models import ResNet
from FiLM.film_module import FiLMLayer

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class BasicBlockFilmed(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.film1 = FiLMLayer(8, planes)
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.film1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def get_resnet_filmed(pretrained=True):
    model = ResNet(BasicBlockFilmed, [2, 2, 2, 2], num_classes=1000)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/resnet18-f37072fd.pth")
        model.load_state_dict(state_dict, strict=False)
    return model

class DoNothingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

class BackBone(nn.Module):
    def __init__(self, pretrained: bool=True, freeze=True) -> None:
        super().__init__()
        # Create the backbone
        self.backbone = get_resnet_filmed(pretrained=pretrained)
        # Load into gpu
        self.backbone.to("cuda")
        # Get film layers of the backbone
        self.backbone_film_layers: List[nn.Module] = self._get_film_layers()
        self.backbone.fc = DoNothingLayer()
        # Freeze all layers except film layers if requested
        self.freeze = freeze
        self.set_grads()

    def _get_film_layers(self) -> List:
        backbone_layers = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
        backbone_film_layers = []
        for layer in backbone_layers:
            for layer_module in layer.modules():
                if not layer_module._get_name() == "BasicBlockFilmed":
                    continue
                backbone_film_layers += [x for x in layer_module.modules() if x._get_name() == "FiLMLayer"]
        return backbone_film_layers

    def set_grads(self) -> None:
        self.backbone.requires_grad_(not self.freeze)
        for x in self.backbone_film_layers:
            x.requires_grad_(True)

    def set_knowledge(self, knowledge: torch.Tensor) -> None:
        for x in self.backbone_film_layers:
            x.set_knowledge(knowledge)

    def forward(self, x, knowledge):
        self.set_knowledge(knowledge)
        out = self.backbone(x)
        return out

class ResNetFiLMed(nn.Module):
    def __init__(self, backbone: BackBone, n_classes: int) -> None:
        super().__init__()

        # Get backbone
        self.backbone: BackBone = backbone
        # Create classification head
        self.fc0 = nn.Linear(512, n_classes).to("cuda")
        self.fc0.requires_grad_(True)

    def forward(self, x, knowledge):
        out = self.backbone(x, knowledge)
        out = self.fc0(out)
        return out

    def forward_detach(self, x, knowledge):
        out = self.backbone(x, knowledge).detach()
        out = self.fc0(out)
        return out

class ResNetNotFiLMed(nn.Module):
    def __init__(self, backbone: BackBone, n_classes: int) -> None:
        super().__init__()

        # Get backbone
        self.backbone: BackBone = backbone
        # Create classification head
        self.fc0 = nn.Linear(512, n_classes).to("cuda")
        self.fc0.requires_grad_(True)

    def forward(self, x, knowledge=None):
        out = self.backbone(x)
        out = self.fc0(out)
        return out