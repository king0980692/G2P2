import sys


class EarlyStopper:
    def __init__(self, patience, stop_by):
        self.patience = patience
        self.trial_counter = 0
        self.stop_by = stop_by
        # self.best_metric = 0 if stop_by == 'metric+' else -float('inf')
        self.best_metric = float(
            'inf') if stop_by != 'metric+' else -float('inf')

    def __call__(self, es_value):
        """Whether to stop training.

        """
        # if self.stop_by == 'metric+':
        #     es_value *= -1

        if es_value < self.best_metric:  # not good
            self.best_metric = es_value
            self.trial_counter = 0
            return False
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1

            if self.patience != sys.maxsize:
                print(
                    f"[ Early Stop Trial: {self.trial_counter}/{self.patience}]")
            return False
        else:
            return True
