from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from soccer_loader import get_match_datasets
from cnn import Net

import time
import datetime
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='CNN for Soccer Match with player attrib')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--no-shuffle', action='store_true', default=False)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--hunits', type=int, default=1024)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--no-save', action='store_true', default=False)
parser.add_argument('--log-interval', type=int, default=50)

args = parser.parse_args()

for arg in vars(args):
    print("{0:{1}<20} {2}".format(str(arg) + ' ', '-', getattr(args, arg)))
print("\n")

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.shuffle = not args.no_shuffle
args.save = not args.no_save

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

features_path = 'match_wplayer_attrib_features2.pkl'
labels_path = 'match_wplayer_attrib_labels.pkl'

train_dataset, test_dataset = get_match_datasets(features_path, labels_path)
print(len(train_dataset))
print(len(test_dataset))

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, **kwargs)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, **kwargs)

model = Net(29, 3)
if args.cuda:
    torch.cuda.set_device(0)
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, nesterov=True, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    # Initialize batchnorm and dropout layers for training
    model.train()

    # Logging variables
    total_loss = 0

    print("| Training")
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        target.squeeze_(-1)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target, requires_grad=False)

        output = model(data)

        loss = criterion(output, target)

        total_loss += loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print("| {:3.4f}%\tLoss: {:1.5e}".format(100. * batch_idx / len(train_loader), total_loss), end='\r')

    print("| {:3.4f}%\tLoss: {:1.5e}".format(100., total_loss))
    return total_loss


def test(epoch):
    # Initialize batchnorm and dropout layers for testing
    model.eval()

    # Logging variables
    correct = 0

    print("| Testing")
    for batch_idx, (data, target) in enumerate(test_loader, 1):
        target.squeeze_(-1)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = Variable(data, requires_grad=False)

        output = model(data)

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).cpu().sum()

        if batch_idx % args.log_interval == 0:
            print("| {:3.4f}%".format(100. * batch_idx / len(test_loader)), end='\r')

    print("| {:3.4f}%".format(100. * batch_idx / len(test_loader)))

    print('\nTest accuracy: {}/{} ({:.4f}%)'.format(
        correct, len(test_loader.dataset),
        (100. * correct) / len(test_loader.dataset)
    ))

    return correct


if __name__ == '__main__':
    min_val_loss = np.inf
    test_correct = 0
    vt = 0.0
    beta = 0.9

    for epoch in range(1, args.epochs + 1):
        time_start = datetime.datetime.now()

        train(epoch)
        test_correct = min(test_correct, test(epoch))

        time_complete = datetime.datetime.now() - time_start
        time_complete = time_complete.total_seconds()
        print("\nTime to complete epoch {} == {} sec(s)".format(
            epoch, time_complete
        ))

        # Calculated moving average for time to complete epoch
        vt = (beta * vt) + ((1 - beta) * time_complete)
        average_epoch_time = vt / (1 - pow(beta, epoch))
        print("Estimated time left == {}".format(
            str(datetime.timedelta(seconds=average_epoch_time * (args.epochs - epoch)))
        ))

        print("{:=<72}\n".format(""))

    print('\nFinal Test accuracy: {}/{} ({:.4f}%)'.format(
        test_correct, len(test_loader.dataset),
        100. * (float(test_correct) / len(test_loader.dataset))
    ))
