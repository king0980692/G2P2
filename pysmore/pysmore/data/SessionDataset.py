import torch
from tqdm import tqdm, trange
from torch.utils.data import IterableDataset
import random
import numpy as np

try:
    from pysmore.utils import utils
except:
    from utils import utils


class ParrallSessionDataset(IterableDataset):

    def __init__(self, sess_df, batch_size, user_cols,
                 item_cols, time_cols, num_neg=2048,
                 trainRandomOrder=False, cached_size=10000000):

        self.cached_size = cached_size
        self.num_neg = num_neg

        self.trainRandomOrder = trainRandomOrder

        self.batch_size = batch_size

        self.data_items = sess_df[item_cols]

        self.offsetSessions = utils.create_offset_session_idx(
            sess_df, user_cols, time_cols)

        self.num_sess = len(self.offsetSessions)

        self.base_order = utils.create_sess_base_order(
            sess_df, user_cols, time_cols, self.num_sess)

        self.sess_df = sess_df
        self.item_cols = item_cols
        self.num_neg = num_neg
        self.cached_size = cached_size

        self.generate_length, self.pop, self.neg_sam = utils.create_popularity(
            sess_df, item_cols, num_neg, cached_size)
        self.cache_idx = 0

    def __len__(self):
        return len(self.user_list)

    def createSessionIdxArr(self):
        """
        Creating an array to fetch data randomly or sequentially
        """
        if (self.trainRandomOrder):
            sessionIdxArr = np.random.permutation(len(self.offsetSessions) - 1)
        else:
            sessionIdxArr = self.base_order

        return sessionIdxArr

    def __iter__(self):
        dataItems = self.data_items.values
        offsetSessions = self.offsetSessions
        sessionIdxArr = self.createSessionIdxArr()

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = offsetSessions[sessionIdxArr[iters]]
        end = offsetSessions[sessionIdxArr[iters] + 1]
        nSessions = len(offsetSessions) - 1

        finished = False
        finishedMask = (end - start <= 1)
        validMask = (iters < nSessions)

        while not finished:
            minlen = (end - start).min()
            outIdx = dataItems[start]

            for i in range(minlen - 1):
                # Build inputs & targets
                inIdx = outIdx
                outIdx = dataItems[start + i + 1]
                if self.num_neg:
                    if self.cached_size:
                        if (self.cache_idx == self.generate_length):

                            _, __, self.neg_sam = utils.create_popularity(
                                self.sess_df, self.item_cols, self.num_neg, self.cached_size)
                            self.cache_idx = 0

                        sample = self.neg_sam[self.cache_idx]
                        self.cache_idx += 1
                    else:
                        _, __, sample = utils.create_popularity(
                            self.sess_df, self.item_cols, self.num_neg, self.cached_size)
                    y = np.hstack([outIdx, sample])
                else:
                    y = outIdx

                input = torch.LongTensor(inIdx)
                target = torch.LongTensor(y)
                yield input, target, finishedMask, validMask

                finishedMask[:] = False
                validMask[:] = True

            start = start + minlen - 1
            # indicator for the sessions to be terminated
            finishedMask = (end - start <= 1)
            nFinished = finishedMask.sum()
            iters[finishedMask] = maxiter + np.arange(1, nFinished + 1)
            maxiter += nFinished

            # indicator for determining valid batch indices
            validMask = (iters < nSessions)
            nValid = validMask.sum()

            if (nValid == 0):
                finished = True
                break

            iters[~validMask] = 0
            sessions = sessionIdxArr[iters[finishedMask]]
            start[finishedMask] = offsetSessions[sessions]
            end[finishedMask] = offsetSessions[sessions + 1]
            iters = iters[validMask]
            start = start[validMask]
            end = end[validMask]
