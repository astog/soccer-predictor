from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from soccer_loader import SoccerDataset
from mlp import Net

import time
import datetime
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='MLP for Soccer Odds Dataset')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--lr-start', type=float, default=0.001)
parser.add_argument('--lr-end', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--no-shuffle', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hunits', type=int, default=1024)
parser.add_argument('--npasses', type=int, default=8)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--dp-hidden', type=float, default=0.5)
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

LR_decay = (args.lr_end / args.lr_start)**(1. / args.epochs)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_subset = 0.2
dataset_path = 'match_odds_only.pkl'

soccer_dataset = SoccerDataset(dataset_path)
num_examples = len(soccer_dataset)
indices = list(range(num_examples))
split = int(np.floor(test_subset * num_examples))

np.random.shuffle(indices)
train_idx, test_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    dataset=soccer_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
test_loader = torch.utils.data.DataLoader(
    dataset=soccer_dataset, batch_size=args.batch_size, sampler=test_sampler, **kwargs)


model = Net(30, 3)
if args.cuda:
    torch.cuda.set_device(0)
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr_start, weight_decay=args.wd)


def train(epoch):
    # Initialize batchnorm and dropout layers for training
    model.train()

    # Logging variables
    train_batch_count = 0
    train_batch_avg_loss = 0
    train_batch_avg_count = 0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target, requires_grad=False)
        target.data.squeeze_(-1)

        output = model(data)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_batch_count += 1
        train_batch_avg_loss += float(loss)
        train_batch_avg_count += 1

        if batch_idx % args.log_interval == 0:
            print("Epoch: {: <6}\tBatches: {: 7.2f}%\tAverage Batch Loss: {:.6e}".format(
                epoch, 100. * train_batch_count / len(train_loader),
                train_batch_avg_loss / train_batch_avg_count
            ))
            train_batch_avg_loss = 0
            train_batch_avg_count = 0

    if train_batch_avg_count > 0:
        print("Epoch: {: <6}\tBatches: {: 7.2f}%\tAverage Batch Loss: {:.6e}".format(
            epoch, 100. * train_batch_count / len(train_loader),
            train_batch_avg_loss / train_batch_avg_count
        ))


def test(epoch):
    # Initialize batchnorm and dropout layers for testing
    model.eval()

    # Logging variables
    correct = 0
    test_batch_count = 0
    test_batch_avg_loss = 0
    test_batch_avg_count = 0

    for batch_idx, (data, target) in enumerate(test_loader, 1):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
        target.data.squeeze_(-1)

        output = model(data)
        loss = criterion(output, target).data.item()  # sum up batch loss

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_batch_count += 1
        test_batch_avg_loss += float(loss)
        test_batch_avg_count += 1

        if batch_idx % args.log_interval == 0:
            print("Epoch: {: <6}\tBatches: {: 7.2f}%\tAverage Batch Loss: {:.6e}".format(
                epoch, 100. * test_batch_count / len(test_loader),
                test_batch_avg_loss / test_batch_avg_count
            ))
            test_batch_avg_loss = 0
            test_batch_avg_count = 0

    if test_batch_avg_count > 0:
        print("Epoch: {: <6}\tBatches: {: 7.2f}%\tAverage Batch Loss: {:.6e}".format(
            epoch, 100. * test_batch_count / len(test_loader),
            float(loss) / test_batch_avg_count
        ))

    print('\nTest set accuracy: {}/{} ({:.4f}%)'.format(
        correct, len(test_loader) * args.batch_size,
        100. * (float(correct) / (len(test_loader) * args.batch_size))
    ))

    return correct


if __name__ == '__main__':
    print("Training batches:", len(train_loader))
    print("Test batches:", len(test_loader), end='\n\n')
    test_correct = 0
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, LR_decay)
    for epoch in range(1, args.epochs + 1):
        time_start = time.clock()
        train(epoch)
        scheduler.step()

        print("\n{:-<72}".format(""))
        print("Test:\n")
        test_correct = test(epoch)

        time_complete = time.clock() - time_start
        print("\nTime to complete epoch {} == {} sec(s)".format(
            epoch, time_complete
        ))
        print("Estimated time left == {}".format(
            str(datetime.timedelta(seconds=time_complete * (args.epochs - epoch)))
        ))

        print("{:=<72}\n".format(""))

    print('\nFinal Test set accuracy: {}/{} ({:.4f}%)'.format(
        test_correct, len(test_loader) * args.batch_size,
        100. * (float(test_correct) / (len(test_loader) * args.batch_size))
    ))
