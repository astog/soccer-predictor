import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from soccer_loader import SoccerDataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


test_subset = 0.2
batch_size = 100
dataset_path = 'match_odds_only.pkl'

soccer_dataset = SoccerDataset(dataset_path)
num_examples = len(soccer_dataset)
indices = list(range(num_examples))
split = int(np.floor(test_subset * num_examples))

np.random.shuffle(indices)
train_idx, test_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = torch.utils.data.DataLoader(
    dataset=soccer_dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(
    dataset=soccer_dataset, batch_size=batch_size, sampler=test_sampler)


class Net(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super(Net, self).__init__()
        self.input_features = input_features
        self.hidden_units = hidden_units

        self.dense1 = nn.Linear(input_features, self.hidden_units)
        self.bn1 = nn.BatchNorm1d(self.hidden_units)

        self.dense2 = nn.Linear(self.hidden_units, self.hidden_units)
        self.bn2 = nn.BatchNorm1d(self.hidden_units)

        self.dense3 = nn.Linear(self.hidden_units, self.hidden_units)
        self.bn3 = nn.BatchNorm1d(self.hidden_units)

        self.dense4 = nn.Linear(self.hidden_units, output_features)

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


model = Net(30, 3, 50)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        target = torch.squeeze(target, dim=-1)
        data, target = Variable(data), Variable(target)

        output = model(data)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if batch_idx % 10 == 0:
            print("[{: >5}, {: >5}, {:.5f}]".format(epoch, len(train_loader) - batch_idx, loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader, 1):
        target = torch.squeeze(target, dim=-1)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        # sum up batch loss
        loss = criterion(output, target).data[0]
        test_loss += loss

        # Prediciton is max of logits
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        if batch_idx % 10 == 0:
            print("[{: >5}, {: >5}, {:.5f}]".format(epoch, len(train_loader) - batch_idx, loss))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct


if __name__ == '__main__':
    max_correct = 0
    for epoch in range(1, 100 + 1):
        train(epoch)
        test()
