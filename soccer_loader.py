from __future__ import print_function
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd


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

        # Convert to tensor during creation, so indexing is fast operation
        self.features = torch.FloatTensor(self.df[self.input_features].values)
        self.label = torch.LongTensor(self.df[self.output_features].values)

    def __getitem__(self, index):
        features = self.features[index]
        label = self.label[index]
        return (features, label)

    def __len__(self):
        return self.df.shape[0]


def get_match_datasets(feature_path, label_path):
    features = torch.load(feature_path)
    labels = torch.load(label_path)

    features = features.view(features.shape[0], features.shape[1] * features.shape[2])
    features.shape[1]
    means = features.mean(dim=1).unsqueeze(-1)
    std = features.std(dim=1).unsqueeze(-1)
    print(features.shape)
    print(means.shape)
    features.sub_(means)
    features.div_(std)

    test_pcent = 0.2
    idx = int(features.shape[0] * test_pcent)
    return MatchDataset(features[idx:], labels[idx:]), MatchDataset(features[:idx], labels[:idx])


class MatchDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.clone()
        self.labels = labels.clone()

    def __getitem__(self, index):
        features = self.features[index]
        label = self.labels[index]
        return (features, label)

    def __len__(self):
        return self.features.shape[0]
