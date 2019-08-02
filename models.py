## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # input 224 ->  -> 224
        self.conv1_1 = nn.Conv2d(1, 32, 5, padding=2)
        # input 224 ->  -> 112
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.batch_norm_1 = nn.BatchNorm2d(32)

        # input 112 ->  -> 112
        self.conv2_1 = nn.Conv2d(32, 64, 5, padding=2)
        # input 112 ->  -> 56
        self.pool_2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(64, 64, 5, padding=2)
        self.pool_3 = nn.MaxPool2d(2, 2)


        # self.batch_norm_2 = nn.BatchNorm2d(64)


        self.fc1 = nn.Linear(28*28*64, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 136)

        self.dropout = nn.Dropout(p=0.5)



        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        

        x = F.relu(self.conv1_1(x))
        # print(x.shape)
        x = self.pool_1(x)
        x = self.batch_norm_1(x)
        # print(x.shape)


        x = F.relu(self.conv2_1(x))
        # print(x.shape)
        x = self.pool_2(x)

        x = F.relu(self.conv3_1(x))
        x = self.pool_3(x)
        # print(x.shape)
        x = self.dropout(x)


        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        

        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
