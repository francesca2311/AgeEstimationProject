import torch

"""
Nota bene: Funziona soltanto su batch bilanciati
"""
def AAR(n_classes: int):
    def aar(y_real: torch.Tensor, y_pred: torch.Tensor):
        # y_real and y_pred must be expressed as number. Ex. 30
        
        mae = torch.mean(torch.abs(y_pred - y_real))
        maej = []
        for c_real, c_pred in zip(y_real.chunk(n_classes), y_pred.chunk(n_classes)):
            maej.append(torch.mean(torch.abs(c_pred - c_real)))
        maej = torch.tensor(maej, requires_grad=True)
        mmae = torch.mean(maej)
        sigma = torch.sqrt(torch.mean(torch.square(maej - mae)))
        return torch.max(torch.tensor(0), 5-mmae) + torch.max(torch.tensor(0), 5-sigma)
    return aar


def AAR_old(n_classes: int):
    def aar(y_real: torch.Tensor, y_pred: torch.Tensor):
        # y_real and y_pred must be expressed as number. Ex. 30
        mae = torch.mean(torch.abs(y_pred - y_real))
        maej = []
        for c_real, c_pred in zip(y_real.chunk(n_classes), y_pred.chunk(n_classes)):
            maej.append(torch.mean(torch.abs(c_pred - c_real)))
        maej = torch.tensor(maej, requires_grad=True)
        sigma = torch.sqrt(torch.mean(torch.square(maej - mae)))
        return torch.max(torch.tensor(0), 7-mae) + torch.max(torch.tensor(0), 3-sigma)
    return aar


def AAR_no_max(n_classes: int):
    def aar(y_real: torch.Tensor, y_pred: torch.Tensor):
        # y_real and y_pred must be expressed as number. Ex. 30
        mae = torch.mean(torch.abs(y_pred - y_real))
        maej = []
        for c_real, c_pred in zip(y_real.chunk(n_classes), y_pred.chunk(n_classes)):
            maej.append(torch.mean(torch.abs(c_pred - c_real)))
        maej = torch.tensor(maej, requires_grad=True)
        mmae = torch.mean(maej)
        sigma = torch.sqrt(torch.mean(torch.square(maej - mae)))
        return mmae + sigma
    return aar


def MMAE(n_classes: int):
    def aar(y_real: torch.Tensor, y_pred: torch.Tensor):
        # y_real and y_pred must be expressed as number. Ex. 30
        mae = torch.mean(torch.abs(y_pred - y_real))
        maej = []
        for c_real, c_pred in zip(y_real.chunk(n_classes), y_pred.chunk(n_classes)):
            maej.append(torch.mean(torch.abs(c_pred - c_real)))
        maej = torch.tensor(maej, requires_grad=True)
        mmae = torch.mean(maej)
        return mmae
    return aar


def SIGMA(n_classes: int):
    def aar(y_real: torch.Tensor, y_pred: torch.Tensor):
        # y_real and y_pred must be expressed as number. Ex. 30
        mae = torch.mean(torch.abs(y_pred - y_real))
        maej = []
        for c_real, c_pred in zip(y_real.chunk(n_classes), y_pred.chunk(n_classes)):
            maej.append(torch.mean(torch.abs(c_pred - c_real)))
        maej = torch.tensor(maej, requires_grad=True)
        sigma = torch.sqrt(torch.mean(torch.square(maej - mae)))
        return sigma
    return aar