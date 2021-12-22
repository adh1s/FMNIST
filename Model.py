import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

#https://madebyollin.github.io/convnet-calculator/ - for calculating sizes of conv outputs

class CNNclassifier(nn.Module):
  def __init__(self, img_size: int, n_channels: int, n_classes: int, dropout_p: float = 0.5):
    super().__init__()
    # parameters of dataset
    self.img_size = img_size
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.dropout_p = dropout_p

    # define layers
    self.conv1 = nn.Sequential(
          nn.Conv2d(self.n_channels,16,3, padding='same'),
          nn.BatchNorm2d(16),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    self.conv2 = nn.Sequential(
          nn.Conv2d(16,32,3, padding='same'),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2)
        )

    self.fc1 = nn.Linear(in_features=int(32*self.img_size/4*self.img_size/4), out_features=512)
    self.bn1 = nn.BatchNorm1d(512)
    self.fc2 = nn.Linear(in_features=512, out_features=128)
    self.bn2 = nn.BatchNorm1d(128)
    self.fc3 = nn.Linear(in_features=128, out_features=self.n_classes)
    self.dropout_layer = nn.Dropout(p=self.dropout_p)

  # define forward function
  def forward(self, x): # 1 x 28 x 28 (C * H * W)
    # conv 1
    x = self.conv1(x) #16 x 14 x 14
    # conv 2
    x = self.conv2(x) #32 x 7 x 7
    # flatten
    x = torch.flatten(x, 1) #1 x 1568
    # fc1
    x = F.relu(self.dropout_layer(self.bn1(self.fc1(x)))) #1 x 512
    # fc2
    x = F.relu(self.dropout_layer(self.bn2(self.fc2(x)))) #1 x 128
    # fc3
    x = self.fc3(x) #1 x 10
    return x #don't need softmax here since loss function used is cross-entropy

'''
#model parameters for F-MNIST

net = CNNclassifier(img_size=28, n_channels=1, n_classes=10)
summary(net, input_size=(1, 28, 28))

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
       BatchNorm2d-2           [-1, 16, 28, 28]              32
              ReLU-3           [-1, 16, 28, 28]               0
         MaxPool2d-4           [-1, 16, 14, 14]               0
            Conv2d-5           [-1, 32, 14, 14]           4,640
       BatchNorm2d-6           [-1, 32, 14, 14]              64
              ReLU-7           [-1, 32, 14, 14]               0
         MaxPool2d-8             [-1, 32, 7, 7]               0
            Linear-9                  [-1, 512]         803,328
      BatchNorm1d-10                  [-1, 512]           1,024
          Dropout-11                  [-1, 512]               0
           Linear-12                  [-1, 128]          65,664
      BatchNorm1d-13                  [-1, 128]             256
          Dropout-14                  [-1, 128]               0
           Linear-15                   [-1, 10]           1,290
================================================================
Total params: 876,458
Trainable params: 876,458
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.48
Params size (MB): 3.34
Estimated Total Size (MB): 3.83

'''
