import torch
import torch.nn as nn
from models import loss as pysmore_loss

loss_function_to_dtype = {
    nn.BCELoss: torch.float32,
    nn.MSELoss: torch.float32,
    nn.CrossEntropyLoss: torch.long,
    nn.NLLLoss: torch.long,
    nn.BCEWithLogitsLoss: torch.float32,
    pysmore_loss.SMSE: torch.float32
}


def convert_target_dtype(target, loss_function):
    target_dtype = loss_function_to_dtype.get(
        type(loss_function), target.dtype)
    return target.to(target_dtype)
