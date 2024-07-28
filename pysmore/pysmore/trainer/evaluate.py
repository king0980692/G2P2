import numpy as np
import torch
import torch.nn as nn
from .TrainerUtils import convert_target_dtype


def eval_with_metric(metric_fns, predictions, answers, verbose=False):
    metric_dict = {}
    for metric in metric_fns:
        metric_name = metric.__name__
        metric_val = metric(predictions, answers)

        if verbose:
            print("\t↳ {:8} {:>8.6f} ".format(metric_name, metric_val))
        metric_dict[metric_name] = metric_val
    return metric_dict


def evaluate(
    metric_fns, model, device, loss_cls, val_dl, epoch_i=0, writer=None, debug=False
):
    pbar = val_dl

    predictions, answers = [], []
    loss_list = []
    total_samlpes = 0

    model.eval()
    with torch.no_grad():
        for id, (x_dict, targets) in enumerate(pbar):
            features = {k: v.to(device) for k, v in x_dict.items()}

            targets = targets.to(device)
            targets = convert_target_dtype(targets, loss_cls)
            preds = model(features)

            loss = loss_cls(preds, targets)  # criterion

            loss_list.append(loss.item())
            total_samlpes += len(targets)

            predictions += (
                preds.argmax(1).tolist() if len(preds.shape) > 1 else preds.tolist()
            )
            answers += targets.tolist()
    predictions = torch.tensor(predictions)
    answers = torch.tensor(answers)

    metric_dict = eval_with_metric(metric_fns, predictions, answers)

    if writer:
        writer.add_scalar("Loss/Val", loss.item(), epoch_i * len(pbar) + id)
    # val_loss = np.mean(loss_list)
    if writer:
        writer.flush()

    # return loss_list[-1], metric_dict, x_dict

    return np.mean(loss_list), metric_dict, x_dict
