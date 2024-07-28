import optuna
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter


from .callback import EarlyStopper
from .TrainerArgs import TrainingArgs
from .TrainerUtils import convert_target_dtype
from .evaluate import evaluate, eval_with_metric


class FeatureWisedTrainer:
    def __init__(
        self,
        TrainingArgs,
        device,
        model,
        loss_cls,
        optimizer,
        scheduler,
        train_dataset,
        eval_dataset,
        metric_fns=None,
        tokernizer=None,
        data_collator=None,
        optuna_trial=None,
    ):
        """
        Notes:
            FeatureWised trainer using Pointwise Loss
        """
        self.device = device
        self.model = model
        self.loss_cls = loss_cls
        self.metric_fns = metric_fns if metric_fns is not None else []
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.train_args = TrainingArgs

        self.writer = SummaryWriter()
        # self.writer = None
        self.early_stop_by = self.train_args.es_by
        self.early_stopper = EarlyStopper(
            self.train_args.es_patience, self.early_stop_by
        )
        self.trial = optuna_trial
        # self.optim = optim.Adam([{'params': self.model.token_embedding.weight},
        # {'params': self.model.positional_embedding},
        # {'params': self.model.transformer.parameters()},
        # ], lr=args.lr)

    def _train_epoch(self, epoch_i, train_dataset, log_interval=5):
        """
        Train a step, a.k.a train an epoch,
        which means model will iterate whole train instances with mini-batches.
        """

        total_len = len(train_dataset)
        log_interval = total_len // log_interval - 1

        if log_interval <= 0:
            log_interval = 1

        if not self.train_args.silence:
            pbar = tqdm(
                enumerate(train_dataset),
                desc="Epoch {}".format(epoch_i),
                total=total_len,
            )
        else:
            pbar = enumerate(train_dataset)

        predictions, answers = [], []
        loss_list = []
        self.model.train()
        start_time = time.time()
        for id, (x_dict, targets) in pbar:
            features = {k: v.to(self.device) for k, v in x_dict.items()}

            # targets = torch.tensor(list(map(lambda x: int(x), targets)))
            targets = convert_target_dtype(targets, self.loss_cls)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            preds = self.model(features)
            loss = self.loss_cls(preds, targets)  # criterion

            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            # loss_list += loss.cpu().tolist()

            predictions += (
                preds.argmax(1).tolist() if len(preds.shape) > 1 else preds.tolist()
            )
            answers += targets.tolist()

            """
            if (id > 0 and id % log_interval == 0):
                # elapsed = time.time() - start_time
                print(
                    "\n[{:>5d}/{:>5d} batches ] - Train_eval"
                    .format(
                        id, total_len-1
                    )
                )

                eval_with_metric(self.metric_fns, predictions,
                                 answers, verbose=True)
                start_time = time.time()
            """

        # train_loss = np.mean(loss_list)
        if self.writer:
            self.writer.add_scalar("Loss/Train", loss.item(), epoch_i * total_len + id)

        # if self.writer:
        #     self.writer.flush()

        return np.mean(loss_list), features
        # return loss_list[-1], features

    def fit(self):
        self.model = self.model.to(self.device)
        best_score = sys.maxsize
        s_time = time.time()
        x_dict = None
        for epoch_i in range(self.train_args.n_train_epochs):
            epoch_start_time = time.time()
            (epoch_train_loss, x_dict) = self._train_epoch(
                epoch_i, self.train_dataset, log_interval=self.train_args.log_interval
            )
            epoch_end_time = time.time()

            if self.scheduler is not None:
                # if epoch_i % self.scheduler.step_size == 0:
                # print("Current lr : {}".format(self.optimizer. state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  # update lr in epoch level by scheduler

            if self.eval_dataset is not None:
                epoch_valid_loss, metric_dict, x = evaluate(
                    self.metric_fns,
                    self.model,
                    self.device,
                    self.loss_cls,
                    self.eval_dataset,
                    epoch_i,
                    self.writer,
                    debug=False,
                )

                if self.trial is not None:
                    self.trial.report(epoch_valid_loss, epoch_i)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()

                if not self.train_args.silence:
                    print("=" * 59)

                    print(
                        "[Elapsed Time: {:5.2f}s ] - Valid_eval".format(
                            epoch_end_time - epoch_start_time
                        )
                    )

                    # print the loss with red color
                    print(
                        "\n[ \033[92mTrain Loss: {:8.6f}\033[0m /".format(
                            epoch_train_loss
                        ),
                        end="",
                    )

                    # print the val loss with green color
                    print(
                        " \033[91mValid Loss: {:8.6f}\033[0m ]".format(epoch_valid_loss)
                    )

                    for metric_name, metric_val in metric_dict.items():
                        print("\t↳ {:8} {:8.6f}".format(metric_name, metric_val))
                    print("=" * 59)

                if list(metric_dict.values())[0] < best_score:
                    best_score = list(metric_dict.values())[0]

                    if not self.train_args.silence:
                        print(
                            "Save best model into {}".format(self.train_args.output_dir)
                        )

                    torch.save(self.model, self.train_args.output_dir)

                # early stopping
                es_by = (
                    epoch_valid_loss
                    if self.early_stop_by == "loss"
                    else list(metric_dict.values())[0]
                )

                if self.early_stopper(es_by) and not self.train_args.silence:
                    print(
                        "\n\n==== Early Stop trigged by {}  ====".format(
                            self.early_stop_by
                        )
                    )
                    break

        elapsed_time = time.time() - s_time
        if not self.train_args.silence:
            print("\n[ Total Time: {:5.2f}s ]".format(elapsed_time))
        if self.writer:
            self.writer.close()
        return self.model, x_dict
