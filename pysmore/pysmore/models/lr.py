import sys
from torch.optim.lr_scheduler import _LRScheduler


class LRPolicyScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, update_steps, total_steps, decay_interval):
        self.base_lr = base_lr
        self.last_lr = [base_lr]
        self.min_lr = 0.0001*self.base_lr
        self.decay_interval = decay_interval
        self.update_steps = update_steps
        self.total_steps = total_steps
        self.step_count = 0
        self.count = 1

        super(LRPolicyScheduler, self).__init__(optimizer)

    def get_lr(self):
        self.step_count += self.update_steps
        # if self.step_count < self.total_steps:
        # # warmup
        # scale = 1.0 - (self.step_count - self.total_steps) / self.total_steps
        # lr = [self.base_lr * scale for base_lr in self.base_lrs]
        # self.last_lr = lr
        # elif self.step_count < self.decay_start_step:
        # decay
        # if step_count < self.num_warmup_steps:
        # # warmup
        # scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
        # lr = [base_lr * scale for base_lr in self.base_lrs]
        # self.last_lr = lr
        # elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
        # # decay
        # decayed_steps = step_count - self.decay_start_step
        # scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
        # min_lr = 0.0001 * self.init_lr
        # lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
        # self.last_lr = lr
        if self.step_count / self.decay_interval > self.count:
            lr = self.base_lr * (1 - (self.step_count / self.total_steps))
            # print(lr, self.step_count, self.total_steps, self.decay_interval)
            lr = self.min_lr if lr < self.min_lr else lr
            self.last_lr = [lr]
            self.count += 1

        # else:
            # if self.num_decay_steps > 0:
            # # freeze at last, either because we're after decay
            # # or because we're between warmup and decay
            # lr = self.last_lr
            # else:
            # do not adjust
            # lr = self.base_lrs
            # lr = self.last_lr
        return self.last_lr
