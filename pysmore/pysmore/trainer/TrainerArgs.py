from typing import Optional
from dataclasses import dataclass


@dataclass
class TrainingArgs:
    output_dir: str

    train_batch_size: Optional[int] = 16
    eval_batch_size: Optional[int] = 16
    n_train_epochs: Optional[int] = 1

    weight_decay: Optional[float] = 0.01
    learning_rate: Optional[float] = 0.025
    n_targets: Optional[int] = 1

    es_patience: Optional[int] = 5
    es_by: Optional[str] = 'loss'
    log_interval: Optional[int] = 10

    silence: Optional[bool] = True

    # save_strategy
    # eval_strategy
