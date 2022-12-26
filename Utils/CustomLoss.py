import torch


def LDAE_loss(y_pred, y_real):
    """
    From ... .
    Note: it leads to Nan, needs a re-check of the code
    """
    return -torch.mean(torch.sum(y_real * torch.log(y_pred) + (1-y_real)*torch.log(1-y_pred), axis=-1))