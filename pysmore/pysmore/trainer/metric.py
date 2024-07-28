import sys
import inspect
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# from sklearn.metrics import *


def AUC(preds, answers):
    return roc_auc_score(answers, preds)


def Accuracy(preds, answers):
    return accuracy_score(answers, preds)


def M_F1(preds, answers):
    return f1_score(answers, preds, average='macro')


def u_F1(preds, answers):
    return f1_score(answers, preds, average='micro')


def MAE(preds, answers):
    return mean_absolute_error(answers, preds)


def MSE(preds, answers):
    return mean_squared_error(answers, preds)


def RMSE(preds, answers):
    return mean_squared_error(answers, preds, squared=False)


def R2(preds, answers):
    return r2_score(answers, preds)


# __all__ = [
    # 'AUC',
    # 'Accuracy',
    # 'M_F1',
    # 'u_F1',
    # 'MSE',
# ]

# 假設 current_module 是當前模塊的名稱
current_module = globals()['__name__']

__all__ = [name for name, obj in inspect.getmembers(sys.modules[current_module])
           if inspect.isfunction(obj) and obj.__module__ == current_module]
