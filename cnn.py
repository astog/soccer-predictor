import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, channels_in, num_classes):
        super(Net, self).__init__()
        '''

        self.conv1 = nn.Conv2d(channels_in, 64, 3)
        self.conv2 = nn.Conv2d(64, 512, 5)

        self.fc1 = nn.Linear(512 * 1 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 3)
        x = F.relu(x)

        x = x.view(-1, 512 * 1 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return(x)

        '''
        self.channels_in = channels_in
        self.fc1 = nn.Linear(20*channels_in, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 3)

    def forward(self, x):
        x = x.view(-1, 20*self.channels_in)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return(x)
