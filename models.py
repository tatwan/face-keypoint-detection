## Here we define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        # M = 224X224 , K = 2X2
        # First Activation/Feature Map = (224-5)/1 + 1 = 220 so output is (32 X 220 X 220)
       
        # maxpool that uses a square window of kernel_size=2, stride=2
      
        self.maxpool1 = nn.MaxPool2d(2,2)
        # After MaxPool new size = (32 X 110 X 110)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(2,2)
        
        #M = 220X220, K = 2X2
        # Second Activation/Feature Map = (110-3)/1 + 1 = 108 so output is (64 X 108 X 108)
        # After MaxPool the new size becomes (64 X 54 X 54)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(2,2) 
        
        # Third Activation/Feature Map = (54 -3)/1 + 1 = 52 so output is (128 x 52 x 52)
        # After MaxPool the new size becomes (128, 26, 26)
        # Adding a Batch Normalization layer
        self.batchnorm = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.maxpool4 = nn.MaxPool2d(2,2) 
        # Fourth Activation/Feature Map = (26 -3)/1 + 1 = 24 so output is (256 x 24 x 24)
        # After MaxPool the new size becomes (256, 12, 12)
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.maxpool5 = nn.MaxPool2d(2,2) 
        # Fourth Activation/Feature Map = (12 - 1)/1 + 1 = 12 so output is (512 x 12 x 12)
        # After MaxPool the new size becomes (512, 6, 6)
        
        
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        
        # dropout with p=0.4
        self.fc_drop = nn.Dropout(p=0.4)
       

        # finally, create 136 output channels for the 68 keypoints
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 136)

        

        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:

        # Five Convolutinal Layeres with MaxPooling of 2X2. No Zero Padding. Stride is 1.
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.maxpool3(F.relu(self.conv3(x)))
        x = self.batchnorm(self.maxpool4(F.relu(self.conv4(x))))
        x = self.maxpool5(F.relu(self.conv5(x)))
        
        # prep for linear layer by flattening the feature maps into feature vectors
        x = x.view(x.size(0), -1)
        
        # Four Fully Connected (FC) linear layers with dropout in between FC1/FC2 and FC2/FC3
        # two conv/relu + pool layers
        
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc_drop(x)
        x = F.relu(self.fc3(x))
#         x = self.fc_drop(x)
        x = self.fc4(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        
        return x
