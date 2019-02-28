import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class model2d(nn.Module):
    def __init__(self, im_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(model2d, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        channels, H, W = im_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3,3))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((3,4),(3,4)))
        self.fc = nn.Linear(12*16*32, 2)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass.
        #############################################################################
        scores = self.layer1(images)
        scores = self.layer2(scores)
        scores = scores.view(scores.size(0), -1)
        scores = self.fc(scores)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores
