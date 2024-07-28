import torch
import torch.nn as nn
import torch.nn.functional as F


class BPR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos, neg):
        x_uij = pos - neg
        log_prob = F.logsigmoid(x_uij).sum()

        return -log_prob


class MaxMargin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos, neg, margin=1.0):
        return F.margin_ranking_loss(
            pos, neg, torch.ones_like(pos),
            margin=margin
        )


class SMSE(nn.Module):
    """
    Sum of Squared Errors
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return F.mse_loss(pred, target, reduction='sum')


class SSE(nn.Module):
    """
    Sum of Squared Errors
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return torch.sum(torch.pow((pred - target), 2))
