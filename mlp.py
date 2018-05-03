import torch.nn as nn
import torch.nn.functional as F


net_config = [30, 60, 30]


class Net(nn.Module):
    def __init__(self, input_features, output_features):
        super(Net, self).__init__()
        self.input_features = input_features

        self.dense1 = nn.Linear(input_features, net_config[0])
        self.bn1 = nn.BatchNorm1d(net_config[0])

        self.dense2 = nn.Linear(net_config[0], net_config[1])
        self.bn2 = nn.BatchNorm1d(net_config[1])

        self.dense3 = nn.Linear(net_config[1], net_config[2])
        self.bn3 = nn.BatchNorm1d(net_config[2])

        self.dense4 = nn.Linear(net_config[2], output_features)

    def forward(self, x):
        # Input layer
        x = x.view(-1, self.input_features)

        x = F.relu(self.dense1(x))
        x = self.bn1(x)

        x = F.relu(self.dense2(x))
        x = self.bn2(x)

        x = F.relu(self.dense3(x))
        x = self.bn3(x)

        # Output Layer (no softmax because the way pytorch crossentropy loss works)
        x = self.dense4(x)
        return x
