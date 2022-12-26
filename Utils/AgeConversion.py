import torch


def EVAge(y):
    return torch.sum(torch.arange(y.shape[-1]).repeat(len(y), 1) * y.to("cpu"), axis=-1)

def ArgMaxAge(y):
    return torch.argmax(y, axis=-1).float()