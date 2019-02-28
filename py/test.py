from __future__ import print_function
import argparse
import os
from PIL import Image
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
parser.add_argument('--testfile', type=str,
                    help='file name for testing')
parser.add_argument('--labelfile', type=str,
                    help='file name for labels')
parser.add_argument('--batch-size', type=int, metavar='N', default=1,
                    help='input batch size for testing')
parser.add_argument('--model',
                    choices=['model2d'],
                    help='which model to train/evaluate')
parser.add_argument('--weights', type=str,
                    help='which model weights to use')
parser.add_argument('--no-labels', action='store_true', default=False,
                    help='outputs the predictions only')
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-frames', action='store_true', default=False,
                    help='save prediction frames')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}


model = torch.load('weights/'+args.weights+'.pt')
if args.cuda:
    model.cuda()

# cross-entropy loss function
criterion = F.cross_entropy


def loader(dataset,label,batch_size):
    size = dataset.shape[0]
    # indices = np.random.permutation(size)
    indices = np.arange(size)
    out = []
    for i in range(int(size/batch_size)):
        chop = dataset[indices[i*batch_size:(i+1)*batch_size]]
        chop_label = label[indices[i*batch_size:(i+1)*batch_size]]
        out.append([chop,chop_label])
    return iter(out)


def evaluate(verbose=False):
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    if args.save_frames:
        os.makedirs('predictions/'+args.testfile)
        os.makedirs('predictions/'+args.testfile+'/positive')
        os.makedirs('predictions/'+args.testfile+'/negative')
        pos_counter = 0
        neg_counter = 0
    for X, Y in loader(test_data, test_label, args.batch_size):
        with torch.no_grad():
            images, targets = Variable(torch.tensor(X)), Variable(torch.tensor(Y))
            images, targets = images.float(), targets.long()
            if args.cuda:
                images, targets = images.cuda(), targets.cuda()
            output = model(images)
            loss += criterion(output, targets, size_average=False).item()
            # predict the argmax of the log-probabilities
            pred = output.data.max(1, keepdim=True)[1]
            if args.save_frames:
                images = images.cpu().numpy()
                predictions = pred.cpu().numpy()
                for i in range(len(pred.cpu())):
                    if predictions[i]:
                        for frame in images[i]:
                            im = Image.fromarray(frame)
                            im = im.convert("L")
                            im.save('predictions/'+args.testfile+'/positive/'+str(pos_counter)+'.jpeg')
                            pos_counter += 1
                    else:
                        for frame in images[i]:
                            im = Image.fromarray(frame)
                            im = im.convert("L")
                            im.save('predictions/'+args.testfile+'/negative/'+str(neg_counter)+'.jpeg')
                            neg_counter += 1
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
            n_examples += pred.size(0)

    loss /= n_examples
    acc = 100. * correct / n_examples
    if verbose:
        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, correct, n_examples, acc))
    return loss, acc


def evaluate_nolabel():
    prediction = np.ones(test_data.shape[0])
    model.eval()
    i = 0
    for X in test_data:
        with torch.no_grad():
            images = Variable(torch.tensor(X))
            images = images.float()
            if args.cuda:
                images = images.cuda()
            output = model(images.unsqueeze(0))
            pred = output.data.max(1, keepdim=True)[1]
            prediction[i] = pred
            i += 1
    np.save('predictions/'+args.testfile, prediction)


test_data = np.load('data/test/'+args.testfile+'.npy')
if args.no_labels:
    evaluate_nolabel()
else:
    test_label = np.load('labels/test/'+args.labelfile+'.npy')
    evaluate(verbose=True)
