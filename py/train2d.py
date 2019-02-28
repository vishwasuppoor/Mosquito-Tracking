from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from models import *

# Training settings
parser = argparse.ArgumentParser(description='mosquito project')
# Hyperparameters
parser.add_argument('--trainfile', type=str, metavar='TF',
                    help='file name for training')
parser.add_argument('--labelfile', type=str, metavar='LF',
                    help='file name for labels')
parser.add_argument('--validate', action='store_true', default=False,
                    help='whether or not to evaluate a validation set')
parser.add_argument('--lr', type=float, metavar='LR', default=0.001,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N', default=40,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N', default=10,
                    help='number of epochs to train')
parser.add_argument('--model',
                    choices=['model2d'],
                    help='which model to train/evaluate')
parser.add_argument('--hidden-dim', type=int,
                    help='number of hidden features/activations')
parser.add_argument('--kernel-size', type=int,
                    help='size of convolution kernels/filters')
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--continue-training', action='store_true', default=False,
                    help='continues training')
parser.add_argument('--weights', type=str, default=False,
                    help='which weights to continue training on')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=27, metavar='N',
                    help='input batch size for testing (default: 27)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='number of batches between logging train status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

n_classes = 2
im_size = (5, 108, 192)

if args.model == 'model2d':
    model = model2d(im_size, n_classes)
elif args.continue_training:
    model = torch.load('weights/'+args.weights+'.pt')
if args.cuda:
    model.cuda()
# cross-entropy loss function
criterion = F.cross_entropy
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

train_data = np.load('data/train/'+args.trainfile+'.npy')
train_label = np.load('labels/train/'+args.labelfile+'.npy')


def loader(dataset,label,batch_size):
    size = dataset.shape[0]
    indices = np.random.permutation(size)
    # indices = np.arange(size)
    out = []
    for i in range(int(size/batch_size)):
        chop = dataset[indices[i*batch_size:(i+1)*batch_size]]
        chop_label = label[indices[i*batch_size:(i+1)*batch_size]]
        out.append([chop,chop_label])
    return iter(out)


def train(epoch):
    '''
    Train the model for one epoch.
    '''
    # Some models use slightly different forward passes and train and test
    # time (e.g., any model with Dropout). This puts the model in train mode
    # (as opposed to eval mode) so it knows which one to use.
    model.train()
    total_loss = 0
    n_samples = 0
    # train loop
    for X, Y in loader(train_data,train_label,args.batch_size):
        # prepare data
        images, targets = Variable(torch.tensor(X)), Variable(torch.tensor(Y))
        images, targets = images.float(), targets.long()
        if args.cuda:
            images, targets = images.cuda(), targets.cuda()
        #############################################################################
        # TODO: Update the parameters in model using the optimizer from above.
        # This only requires a couple lines of code.
        #############################################################################
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_samples += outputs.shape[0]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    if epoch % args.log_interval == 0:
        print('Train Epoch: {}\t'
              'Train Loss: {:.6f}'.format(epoch, total_loss/n_samples))


def evaluate(split, verbose=False):
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    if split == 'train':
        loader_data, loader_label, loader_batch_size = train_data, train_label, args.batch_size
    elif split == 'validation':
        loader_data, loader_label, loader_batch_size = validation_data, validation_label, args.batch_size
    for X, Y in loader(loader_data, loader_label, loader_batch_size):
        with torch.no_grad():
            images, targets = Variable(torch.tensor(X)), Variable(torch.tensor(Y))
            images, targets = images.float(), targets.long()
            if args.cuda:
                images, targets = images.cuda(), targets.cuda()
            output = model(images)
            loss += criterion(output, targets, size_average=False).item()
            # predict the argmax of the log-probabilities
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
            n_examples += pred.size(0)

    loss /= n_examples
    acc = 100. * correct / n_examples
    if verbose:
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            split, loss, correct, n_examples, acc))
    return loss, acc


# train the model one epoch at a time
for epoch in range(1, args.epochs + 1):
    train(epoch)

evaluate('train', verbose=True)

if args.validate:
    validation_data = np.load('data/validation/'+args.trainfile+'.npy')
    validation_label = np.load('labels/validation/'+args.labelfile+'.npy')
    evaluate('validation', verbose=True)

# Save the model (architecture and weights)
torch.save(model, 'weights/' + args.trainfile + '_' + args.model + '.pt')