import torch


def AAR(samples_per_class: int):
    def aar(y_real: torch.Tensor, y_pred: torch.Tensor):
        # y_real and y_pred must be expressed as number. Ex. 30
        
        mae = torch.mean(torch.abs(y_pred - y_real))
        maej = []
        for c_real, c_pred in zip(y_real.chunk(samples_per_class), y_pred.chunk(samples_per_class)):
            maej.append(torch.mean(torch.abs(c_pred - c_real)))
        maej = torch.tensor(maej, requires_grad=True)
        mmae = torch.mean(maej)
        sigma = torch.sqrt(torch.mean(torch.square(maej - mae)))
        return torch.max(torch.tensor(0), 5-mmae) + torch.max(torch.tensor(0), 5-sigma)
    return aar


def AAR_old(samples_per_class: int):
    def aar(y_real: torch.Tensor, y_pred: torch.Tensor):
        # y_real and y_pred must be expressed as number. Ex. 30
        mae = torch.mean(torch.abs(y_pred - y_real))
        maej = []
        for c_real, c_pred in zip(y_real.chunk(samples_per_class), y_pred.chunk(samples_per_class)):
            maej.append(torch.mean(torch.abs(c_pred - c_real)))
        maej = torch.tensor(maej, requires_grad=True)
        sigma = torch.sqrt(torch.mean(torch.square(maej - mae)))
        return torch.max(torch.tensor(0), 7-mae) + torch.max(torch.tensor(0), 3-sigma)
    return aar


def AAR_no_max(samples_per_class: int):
    def aar(y_real: torch.Tensor, y_pred: torch.Tensor):
        # y_real and y_pred must be expressed as number. Ex. 30
        mae = torch.mean(torch.abs(y_pred - y_real))
        maej = []
        for c_real, c_pred in zip(y_real.chunk(samples_per_class), y_pred.chunk(samples_per_class)):
            maej.append(torch.mean(torch.abs(c_pred - c_real)))
        maej = torch.tensor(maej, requires_grad=True)
        mmae = torch.mean(maej)
        sigma = torch.sqrt(torch.mean(torch.square(maej - mae)))
        return mmae + sigma
    return aar


def MMAE(samples_per_class: int):
    def aar(y_real: torch.Tensor, y_pred: torch.Tensor):
        # y_real and y_pred must be expressed as number. Ex. 30
        mae = torch.mean(torch.abs(y_pred - y_real))
        maej = []
        for c_real, c_pred in zip(y_real.chunk(samples_per_class), y_pred.chunk(samples_per_class)):
            maej.append(torch.mean(torch.abs(c_pred - c_real)))
        maej = torch.tensor(maej, requires_grad=True)
        mmae = torch.mean(maej)
        return mmae
    return aar


def SIGMA(samples_per_class: int):
    def aar(y_real: torch.Tensor, y_pred: torch.Tensor):
        # y_real and y_pred must be expressed as number. Ex. 30
        mae = torch.mean(torch.abs(y_pred - y_real))
        maej = []
        for c_real, c_pred in zip(y_real.chunk(samples_per_class), y_pred.chunk(samples_per_class)):
            maej.append(torch.mean(torch.abs(c_pred - c_real)))
        maej = torch.tensor(maej, requires_grad=True)
        sigma = torch.sqrt(torch.mean(torch.square(maej - mae)))
        return sigma
    return aar