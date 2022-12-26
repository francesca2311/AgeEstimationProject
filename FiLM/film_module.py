import torch
from torch import nn


class FiLMLayer(nn.Module):
    def __init__(self, dim_knowledge, dim_channel):
        super().__init__()
        self.beta = nn.Linear(dim_knowledge, dim_channel)
        self.gamma = nn.Linear(dim_knowledge, dim_channel)
        self._knowledge = None

    def set_knowledge(self, knowledge: torch.Tensor) -> None:
        self._knowledge = knowledge

    def forward(self, x):
        beta = self.beta(self._knowledge).view(x.size(0), x.size(1), 1, 1)
        gamma = self.gamma(self._knowledge).view(x.size(0), x.size(1), 1, 1)
        x = gamma * x + beta
        return x