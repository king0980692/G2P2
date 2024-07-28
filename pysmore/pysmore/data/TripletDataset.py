from tqdm import tqdm, trange
from torch.utils.data import IterableDataset, get_worker_info
import random

try:
    from pysmore.utils.c_alias_method import AliasTable
    from pysmore.utils import utils
    from pysmore.utils.sampler import SMOReSampler
except:
    from utils.c_alias_method import AliasTable
    from utils import utils
    from utils.sampler import SMOReSampler


class SMOReTripletDataset(IterableDataset):

    def __init__(self, user_list, weight_list, user_size, item_size, num_neg):

        self.user_list = user_list
        self.weight_list = weight_list
        self.user_size = user_size
        self.item_size = item_size

        self.sampler = SMOReSampler(
            self.user_list, self.weight_list, num_neg, self.user_size, self.item_size)
        self.cnt = 0

    def __len__(self):
        return len(self.user_list)

    def __iter__(self):
        return self.sampler.triplet_generator()


class SimpleTripletDataset(IterableDataset):
    def __init__(self, user_list, weight_list, user_size, item_size, num_neg):

        self.user_list = user_list
        self.weight_list = weight_list
        self.user_size = user_size
        self.item_size = item_size

    def __len__(self):
        return self.user_size

    def __iter__(self):
        sample_iterator = self._sample()
        return sample_iterator

    def _sample_neg(self, x):
        while True:
            neg_id = random.randint(0, self.item_size - 1)
            if neg_id not in x:
                return neg_id

    def _sample(self):

        worker_info = get_worker_info()
        seed = worker_info.seed % (2**32)
        random.seed(seed)
        indices = [x for x in range(self.user_size)]

        while True:
            # users = random.sample(indices, 1)
            users = random.randint(0, self.user_size - 1)

            # pos_items = interected_items_df['item_id_idx'].apply(lambda x : random.choice(x)).values
            pos_items = random.choice(self.user_list[users])

            # neg_items = interected_items_df['item_id_idx'].apply(lambda x: sample_neg(x)).values
            neg_items = [self._sample_neg(self.user_list[users])]

            yield users, pos_items, neg_items
