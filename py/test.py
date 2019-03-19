from __future__ import print_function
import argparse
import os
import cv2
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
parser.add_argument('--vidname', type=str,
                    help='original video name')
parser.add_argument('--testfile', type=str,
                    help='file name for testing')
parser.add_argument('--labelfile', type=str,
                    help='file name for labels')
parser.add_argument('--batch-size', type=int, metavar='N', default=1,
                    help='input batch size for testing')
parser.add_argument('--model', choices=['model2d'],
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
parser.add_argument('--qualitative', action='store_true', default=False,
                    help='qualitative analysis')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}


model = torch.load('../weights/'+args.weights+'.pt')
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


def draw_label(img,text,h,w):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    rthickness = cv2.FILLED
    tthickness = 2
    margin = 2
    txt_size = cv2.getTextSize(text, font_face, scale, tthickness)
    position = ((w-txt_size[0][0])/2,h-50)
    end_x = position[0] + txt_size[0][0] + margin
    end_y = position[1] - txt_size[0][1] - margin
    cv2.rectangle(img, position, (end_x, end_y), (255,0,0), rthickness)
    cv2.putText(img, text, position, font_face, scale, (0,255,255), 1, tthickness)


def evaluate(verbose=False):
    model.eval()
    loss = 0
    correct = 0
    total_positive = 0
    total_negative = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    n_examples = 0
    all_predictions = np.empty(shape=[0,2])
    if not os.path.isdir('../predictions/'+args.testfile):
        os.makedirs('../predictions/'+args.testfile)
    if args.qualitative:
        if not args.vidname:
            print('Video name not available. Not saving output videos')
            args.qualitative = False
        if not os.listdir('../Frames/'+args.vidname):
            print('HQ images not available. Not saving output videos')
            args.qualitative = False
    if args.qualitative:
        path = '../Frames/'+args.vidname
        image_predictions = []
        framenumbers = []
        test_img = cv2.imread(path+'/frame0.jpg')
        h,w,c = test_img.shape
        sample_text = 'Mosquito Present'
        txt_size = cv2.getTextSize(sample_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        position = (int((w-txt_size[0][0])/2),h-50)
        end_x = position[0] + txt_size[0][0] + 2
        end_y = position[1] - txt_size[0][1] - 2
        video = cv2.VideoWriter('../predictions/'+args.testfile+'/.avi',cv2.VideoWriter_fourcc(*"MJPG"),5,(w,h))

    for X, Y in loader(test_data, test_label, args.batch_size):
        with torch.no_grad():
            images, targets = Variable(torch.tensor(X)), Variable(torch.tensor(Y[:,0]))
            images, targets = images.float(), targets.long()
            if args.cuda:
                images, targets = images.cuda(), targets.cuda()
            output = model(images)
            loss += criterion(output, targets, size_average=False).item()
            # predict the argmax of the log-probabilities
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
            n_examples += pred.size(0)
            pred = pred.cpu().numpy().flatten()
            total_positive += np.sum(Y[:,0] == 1)
            total_negative += np.sum(Y[:,0] == 0)
            true_positive += np.sum(np.logical_and(pred == 1,Y[:,0] == 1))
            true_negative += np.sum(np.logical_and(pred == 0,Y[:,0] == 0))
            false_positive += np.sum(np.logical_and(pred == 1,Y[:,0] == 0))
            false_negative += np.sum(np.logical_and(pred == 0,Y[:,0] == 1))
            predictions = np.column_stack((pred,Y[:,1])).astype(int)
            all_predictions = np.vstack((all_predictions,predictions))
            if args.qualitative:
                for prediction, index in predictions:
                    if prediction:
                        for i in range(5):
                            framenumbers.append(index*5+i)
                            img = cv2.imread(path+'/frame'+str(index*5+i)+'.jpg')
                            img = cv2.rectangle(img, (position[0]-2,position[1]+2), (end_x, end_y), (255,0,0), cv2.FILLED)
                            img = cv2.putText(img,'Mosquito Present',position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
                            image_predictions.append(img)
                    else:
                        for i in range(5):
                            framenumbers.append(index*5+i)
                            img = cv2.imread(path+'/frame'+str(index*5+i)+'.jpg')
                            img = cv2.rectangle(img, (position[0]-2,position[1]+2), (end_x, end_y), (255,0,0), cv2.FILLED)
                            img = cv2.putText(img,'Mosquito Absent',(position[0]+8,position[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                            image_predictions.append(img)
    if args.qualitative:
        sorted_indices = np.argsort(np.array(framenumbers))
        sorted_images = [image_predictions[i] for i in sorted_indices]
        for img in sorted_images:
            video.write(img)
        cv2.destroyAllWindows()
        video.release()

    loss /= n_examples
    acc = 100. * correct / n_examples
    tp = 100. * true_positive / total_positive
    tn = 100. * true_negative / total_negative
    fp = 100. * false_positive / total_negative
    fn = 100. * false_negative / total_positive
    np.save('../predictions/'+args.testfile+'/prediction',all_predictions)
    if verbose:
        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\nTrue Positive: {}/{} ({:.0f}%)\nTrue Negative: {}/{} ({:.0f}%)\n'
            'False Positive: {}/{} ({:.0f}%)\nFalse Negative: {}/{} ({:.0f}%)\n'
            .format(loss, correct, n_examples, acc, true_positive, n_examples, tp, true_negative, n_examples, tn,
                false_positive, n_examples, fp, false_negative, n_examples, fn))
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
    np.save('../predictions/'+args.testfile+'/prediction', prediction)


test_data = np.load('../data/test/'+args.testfile+'.npy')
if args.no_labels:
    evaluate_nolabel()
else:
    test_label = np.load('../labels/test/'+args.labelfile+'.npy')
    evaluate(verbose=True)
