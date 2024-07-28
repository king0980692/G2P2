# import time
from operator import is_
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import csr_matrix


try:
    from pysmore.utils.utils import *
    from pysmore.data.TripletDataset import *
    from pysmore.data.SessionDataset import *
except:
    from utils.utils import *
    from data.TripletDataset import *
    from data.SessionDataset import *


@dataclass
class SessCollator:
    def __call__(self, batches):
        """
        batches:
            list of ()
        """
        import IPython

        IPython.embed(colors="linux")
        exit(1)
        Xs, Y = zip(*batches)
        Xs = np.array(Xs)
        return (
            {name: torch.tensor(Xs[:, id]) for id, name in enumerate(self.feat_names)},
            torch.tensor(Y),
        )


@dataclass
class FeatureCollator:
    feat_names: str

    def __call__(self, batches):
        """
        batches:
            list of ()
        """

        Xs, Y = zip(*batches)
        Xs = np.array(Xs)
        return (
            {name: torch.tensor(Xs[:, id]) for id, name in enumerate(self.feat_names)},
            torch.tensor(Y),
        )


@dataclass
class DynamicSequentialCollator:
    ctx_length: int

    def __call__(self, batches):
        """
        batches:
            list of ()
        """

        fea_names = list(batches[0][0].keys())
        seq_idxs = [id for id, n in enumerate(fea_names) if "s" in n]

        max_len = (
            self.ctx_length
            if self.ctx_length > 0
            else max(map(lambda x: len(list(x[0].values())[0]), batches))
        )

        targets = torch.tensor([int(b[1]) for b in batches])

        return (
            {
                name: (
                    torch.tensor(list(map(lambda x: x[0][name], batches)))
                    if id not in seq_idxs
                    else torch.stack(
                        [
                            torch.nn.functional.pad(
                                torch.tensor(t),
                                (0, max_len - len(t)),
                                mode="constant",
                                value=0,
                            )
                            for t in [list(b[0].values())[0] for b in batches]
                        ]
                    )
                )
                for id, name in enumerate(fea_names)
            },
            targets,
        )


@dataclass
class InteractionBatchify:
    item_pool: np.array
    user_meta_df: Optional[pd.core.frame.DataFrame] = None
    item_meta_df: Optional[pd.core.frame.DataFrame] = None

    def __call__(self, batches):
        """
        batches: list of user
        """
        num_user = len(batches)
        u_batch = [u for u in batches for _ in range(len(self.item_pool))]
        i_batch = np.tile(self.item_pool, num_user)

        output = {}
        if self.user_meta_df is not None:
            u_meta_batch = self.user_meta_df.loc[u_batch].to_dict("list")
            output.update(u_meta_batch)

        if self.item_meta_df is not None:
            i_meta_batch = self.item_meta_df.loc[i_batch].to_dict("list")
            # j_meta_batch = self.item_meta_df.loc[j_batch].to_dict('list')

            output.update(i_meta_batch)

        if len(output) > 0:  # with meta data
            for key, value in output.items():
                output[key] = torch.tensor(value)

            return output
        else:
            return {"u": torch.tensor(u_batch), "i": torch.tensor(i_batch)}


@dataclass
class NegSampleBatchify:
    user_meta_df: Optional[pd.core.frame.DataFrame] = None
    item_meta_df: Optional[pd.core.frame.DataFrame] = None
    dummy_fields: Optional[str] = None

    def __call__(self, batches):
        """
        batches:
            list of (u, i, [neg_i, neg_i2, ..., neg_ix])
        """

        if self.dummy_fields == "item":
            input = torch.empty(len(batches))
            dummy_tensor = torch.zeros_like(input, dtype=torch.long)

            output = {}

            if self.user_meta_df is not None:
                u_meta_batch = self.user_meta_df.loc[batches].to_dict("list")
                output.update(u_meta_batch)
            if self.item_meta_df is not None:
                i_meta_batch = self.item_meta_df.loc[dummy_tensor].to_dict("list")
                output.update(i_meta_batch)

                for key, value in i_meta_batch.items():
                    output[f"neg-{key}"] = value

            if len(output) > 0:
                for key, value in output.items():
                    output[key] = torch.tensor(value)
                return output
            else:
                return {
                    "u": torch.tensor(batches),
                    "i": dummy_tensor,
                    "neg-i": dummy_tensor,
                }

        elif self.dummy_fields == "user":
            input = torch.empty(len(batches))
            dummy_tensor = torch.zeros_like(input, dtype=torch.long)

            output = {}
            if self.user_meta_df is not None:
                u_meta_batch = self.user_meta_df.loc[dummy_tensor].to_dict("list")
                output.update(u_meta_batch)

            if self.item_meta_df is not None:
                i_meta_batch = self.item_meta_df.loc[batches].to_dict("list")

                output.update(i_meta_batch)

                for key, value in i_meta_batch.items():
                    output[f"neg-{key}"] = value

            if len(output) > 0:
                for key, value in output.items():
                    output[key] = torch.tensor(value)

                return output
            else:
                return {
                    "u": dummy_tensor,
                    "i": torch.tensor(batches),
                    "neg-i": dummy_tensor,
                }

        # --------------

        num_neg = len(batches[0][-1])

        u_batch = []
        i_batch = []
        j_batch = []
        for b in batches:
            u_batch += [b[0]] * num_neg
            i_batch += [b[1]] * num_neg
            j_batch += b[2]

        output = {}
        if self.user_meta_df is not None:
            u_meta_batch = self.user_meta_df.loc[u_batch].to_dict("list")
            output.update(u_meta_batch)

        if self.item_meta_df is not None:
            i_meta_batch = self.item_meta_df.loc[i_batch].to_dict("list")
            j_meta_batch = self.item_meta_df.loc[j_batch].to_dict("list")

            output.update(i_meta_batch)

            for key, value in j_meta_batch.items():
                output[f"neg-{key}"] = value

        if len(output) > 0:  # with meta data
            for key, value in output.items():
                output[key] = torch.tensor(value)

            return output
        else:  # without meta data
            return {
                "u": torch.tensor(u_batch),
                "i": torch.tensor(i_batch),
                "neg-i": torch.tensor(j_batch),
            }


class PandasDataset(Dataset):
    def __init__(self, x, y=None):
        super().__init__()

        self._matrix = None
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        if isinstance(y, pd.Series):
            y = y.to_numpy()

        self.x = x
        if y is not None:
            self.y = y
        else:
            self.y = None

    def sparsfiy(self):
        _row = []
        _col = []
        _rating = []

        for line, t in zip(self.x, self.y):
            u = line[0] - 1
            i = line[1] - 1
            _row.append(u)
            _col.append(i)
            _rating.append(t)

        self._matrix = csr_matrix((_rating, (_row, _col)), shape=(6040, 3952))

        temp = self._matrix.tocoo()
        self.item = list(temp.col.reshape(-1))
        self.user = list(temp.row.reshape(-1))
        self.rate = list(temp.data)

    def __getitem__(self, index):
        if self._matrix is not None:
            return (self.user[index], self.item[index]), self.rate[index]

        x_tensor = self.x[index]
        # x_tensor = {k: torch.tensor(v)
        #             for k, v in self.x.iloc[index].to_dict().items()}

        if self.y is not None:
            y_tensor = self.y[index]
            # if isinstance(index, slice):
            # y_tensor = torch.tensor(
            #     self.y.iloc[index].values)
            #     return (x_tensor, y_tensor)
            # else:
            #     y_tensor = torch.tensor(self.y.iloc[index])
            #     return (x_tensor, y_tensor)
            return (x_tensor, y_tensor)
        else:
            return x_tensor

    def __len__(self):
        return len(self.x)


class DLRMDataGenerator:
    def __init__(
        self,
        inter_df,  # interaction dataframe
        val_df,
        user_cols,
        item_cols,
        sampler_type: str = "weighted",
        num_neg: int = 5,
    ):
        # (train_inter_df,
        # val_inter_df) = train_test_split(inter_df, random_state=42)
        train_inter_df = inter_df

        # 4. Create Graph (Only suitable for SMORe associated task)
        (
            user_list,
            weight_list,
            user_size,
            item_size,
        ) = create_lil_graph_by_interaction_df(inter_df, user_cols, item_cols)

        # 5. Create Triplet dataset
        if sampler_type == "simple":  # TODO: need check
            self.triplet_data = SimpleTripletDataset(
                user_list, weight_list, user_size, item_size, num_neg=num_neg
            )

        elif sampler_type == "weighted":
            self.triplet_data = SMOReTripletDataset(
                user_list, weight_list, user_size, item_size, num_neg=num_neg
            )
        else:
            raise ValueError(f"Sampler: {sampler_type} not supported!")

        self.ans = val_df.groupby(user_cols)[item_cols].apply(list).to_dict()

        self.query_user = val_df[user_cols].unique()  # np.array

        self.item_pool = np.union1d(
            train_inter_df[item_cols].unique(), val_df[item_cols].unique()
        )
        """
        self.ans = train_inter_df.groupby(
            user_cols)[item_cols].apply(list).to_dict()

        self.query_user = train_inter_df[user_cols].unique() # np.array

        self.item_pool = np.union1d(
            train_inter_df[item_cols].unique(), val_df[item_cols].unique())
        """

    def generate_data(
        self, batch_size, num_worker, user_meta_df=None, item_meta_df=None
    ):
        triplet_collator = NegSampleBatchify(user_meta_df, item_meta_df)

        triplet_loader = DataLoader(
            self.triplet_data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_worker,
            collate_fn=triplet_collator,
        )

        inter_collactor = InteractionBatchify(
            self.item_pool, user_meta_df, item_meta_df
        )

        np.random.shuffle(self.query_user)

        inter_dataloader = DataLoader(
            self.query_user,
            batch_size=1,
            shuffle=False,
            num_workers=num_worker,
            collate_fn=inter_collactor,
        )

        return triplet_loader, inter_dataloader, self.ans


class ParallelSessDataGenerator:
    def __init__(
        self,
        train_df,  # interaction dataframe
        user_cols,
        item_cols,
        time_cols,
        batch_size,
        sampler_type,
        num_neg,
    ):
        self.train_df = train_df
        self.parallel_sess_iterator = ParrallSessionDataset(
            train_df, batch_size, user_cols, item_cols, time_cols
        )

    def __len__(self):
        return len(self.train_df)

    def generate_data(self):
        sess_collator = SessCollator()

        train_loader = DataLoader(
            self.parallel_sess_iterator,
            shuffle=False,
            batch_size=1,
            collate_fn=sess_collator,
        )

        return train_loader

        # iterator = self.parallel_sess_iterator.__iter__()
        # return (iterator)


class RetrievalDataGenerator:
    def __init__(
        self,
        inter_df,  # interaction dataframe
        user_cols,
        item_cols,
        sampler_type,
        num_neg,
    ):
        # 4. Create Graph
        (
            user_list,
            weight_list,
            user_size,
            item_size,
        ) = create_lil_graph_by_interaction_df(inter_df, user_cols, item_cols)

        # 5. Create Triplet dataset
        if sampler_type == "simple":  # TODO: need check
            self.triplet_data = SimpleTripletDataset(
                user_list, weight_list, user_size, item_size, num_neg=num_neg
            )

        elif sampler_type == "weighted":
            self.triplet_data = SMOReTripletDataset(
                user_list, weight_list, user_size, item_size, num_neg=num_neg
            )
        else:
            raise ValueError(f"Sampler: {sampler_type} not supported!")

        self.all_user = list(set(inter_df[user_cols].to_numpy()))
        self.all_item = list(set(inter_df[item_cols].to_numpy()))

    def generate_data(
        self, batch_size, num_worker, user_meta_df=None, item_meta_df=None
    ):
        triplet_collator = NegSampleBatchify(user_meta_df, item_meta_df)

        dummy_user_collator = NegSampleBatchify(
            user_meta_df, item_meta_df, dummy_fields="item"
        )

        dummy_item_collator = NegSampleBatchify(
            user_meta_df, item_meta_df, dummy_fields="user"
        )

        triplet_loader = DataLoader(
            self.triplet_data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_worker,
            collate_fn=triplet_collator,
        )
        # ---------------
        # Prepare valid dataloader(all user)
        user_dataloader = DataLoader(
            self.all_user,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_worker,
            collate_fn=dummy_user_collator,
        )

        # Prepare valid dataloader(all item)
        item_dataloader = DataLoader(
            self.all_item,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_worker,
            collate_fn=dummy_item_collator,
        )

        return (
            triplet_loader,
            user_dataloader,
            item_dataloader,
        )


class RankingDataGenerator:
    def __init__(self, x, y):
        self.feat_names = x.columns.to_list()
        self.dataset = PandasDataset(x, y)
        self.length = len(self.dataset)
        self.df = pd.concat([x, y], axis=1)

    def generate_data(
        self,
        split_ratio=None,
        split_strategy="random",
        batch_size=1024,
        num_workers=0,
        val_x=None,
        val_y=None,
        test_x=None,
        test_y=None,
        contain_sequence=False,
        context_length=128,
    ):
        assert not (val_x is None) ^ (
            val_y is None
        ), "val_x and val_y should be both None or not None"
        assert not (test_x is None) ^ (
            test_y is None
        ), "test_x and test_y should be both None or not None"

        is_split_flag = False
        if val_x is not None and val_y is not None:
            self.train_data = self.dataset
            self.val_data = PandasDataset(val_x, val_y)
            self.test_data = None

        else:
            is_split_flag = True
            if len(split_ratio) == 2:  # train : val : test
                train_length = int(self.length * split_ratio[0])
                val_length = int(self.length * split_ratio[1])
                test_length = self.length - train_length - val_length
                # print("the instances of train : val are  %d : %d " %
                #       (train_length, val_length))
            else:  # train : val
                train_length = int(self.length * split_ratio[0])
                val_length = self.length - train_length
                test_length = self.length - train_length - val_length
                # print("the instances of train : val : test are  %d : %d : %d" %
                #       (train_length, val_length, test_length))

            if split_strategy == "random":
                (self.train_data, self.val_data, self.test_data) = random_split(
                    self.dataset,
                    (train_length, val_length, test_length),
                    generator=torch.Generator().manual_seed(42),
                )
                """
                (
                    self.train_data,
                    self.val_data,
                ) = train_test_split(self.df,
                                     train_size=split_ratio[0],
                                     random_state=42)
                # self.test_data = None
                """
            else:
                # static split
                self.train_data = Subset(self.dataset, range(train_length))
                self.val_data = Subset(
                    self.dataset, range(train_length, train_length + val_length)
                )
                self.test_data = Subset(
                    self.dataset, range(train_length + val_length, self.length)
                )

        if test_x is not None and test_y is not None:
            self.test_data = PandasDataset(test_x, test_y)

        # --------------------------------

        # for sequence data usage
        data_collator = (
            DynamicSequentialCollator(context_length)
            if contain_sequence
            else FeatureCollator(self.feat_names)
        )

        train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

        val_loader = DataLoader(
            self.val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )
        test_loader = DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

        return (
            train_loader,
            self.train_data,
            val_loader,
            self.val_data,
            test_loader,
            self.test_data,
        )
