from __future__ import print_function
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import numpy as np

test_subset = 0.2
batch_size = 100
dataset_path = 'match_odds_only.pkl'


class SoccerDataset(Dataset):
    def __init__(self, pickle_path):
        """
        Args:
            pickle_path (string): path to pickled file
            input_features (int): number of input features
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.df = pd.read_pickle(pickle_path)
        self.input_features = [
            'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD',
            'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD',
            'WHA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'GBH', 'GBD',
            'GBA', 'BSH', 'BSD', 'BSA'
        ]
        self.output_features = [
            'match_result'
        ]

    def __getitem__(self, index):
        features = self.df[self.input_features].iloc[index].values
        output = self.df[self.output_features].iloc[index].values

        # Make them Float tensor
        x = torch.FloatTensor(features)
        y = torch.LongTensor(output)

        # Return item
        return (x, y)

    def __len__(self):
        return self.df.shape[0]
