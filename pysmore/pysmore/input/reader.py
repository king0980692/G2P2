import itertools
import json
import numpy as np 
import pandas as pd 
import sys

from torch.utils.data import IterableDataset, Dataset
import random


class DatasetReader(object):
    def __init__(self, data_path, field_name, sep='\t'):

        """
        Args:
            data_path: path to the data file
            sep: separator
        """

        self.fpath = data_path
        self.sep = sep
        self.field_name = field_name

    def load(self):
        # Load data

        data_format = list(itertools.chain(*json.loads(self.field_name)))
        target_idx = -1
        try :
            target_idx = data_format.index('y')
        except:
            pass
        try :
            target_idx = data_format.index('r')
        except:
            pass
        assert target_idx != -1, "Target field not found, you must specify target field: y or r"
        target_field = data_format[target_idx]

        df = pd.read_csv(self.fpath,
                         sep=self.sep,
                         engine='python',
                         names=data_format)

        # TODO: Remove negative rating?
        # df = df[df['rate'] >= 3]
        df = df[df[target_field] > 0].reset_index(drop=True)
        return df

                

