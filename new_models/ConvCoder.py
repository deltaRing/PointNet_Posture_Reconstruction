import torch
import torch.nn.functional as F
import numpy as np

class ConvEncoder(torch.nn.Module):
    def __init__(self,
                 batch_size=32,
                 input_size=68,
                 pose_num=1):
        super(ConvEncoder, self).__init__()

        self.ReLU = torch.nn.ReLU()

        self.conv1 = torch.nn.Conv2d(pose_num, 16, 3, 1, 1)
        self.pool1 = torch.nn.MaxPool2d(3, 2, 1, return_indices=True)
        self.conv2 = torch.nn.Conv2d(16, 8, 3, 1, 1)
        self.pool2 = torch.nn.MaxPool2d(3, 2, 1, return_indices=True)

    def forward(self, x):
        x = self.ReLU(self.conv1(x))
        x, pool1 = self.pool1(x)
        x = self.ReLU(self.conv2(x))
        x, pool2 = self.pool2(x)

        return x, [pool1, pool2]


class ConvDecoder(torch.nn.Module):
    def __init__(self,
                 batch_size=32,
                 input_size=12,
                 input_num=8,
                 pose_num=1
                 ):
        super(ConvDecoder, self).__init__()

        self.ReLU    = torch.nn.ReLU()
        self.Sigmoid = torch.nn.Sigmoid()

        self.deconv1 = torch.nn.ConvTranspose2d(input_num, 8, 4, 2, 1)
        self.deconv2 = torch.nn.ConvTranspose2d(8, 16, 4, 2, 1)
        self.deconv3 = torch.nn.ConvTranspose2d(16, pose_num, 3, 1, 1)

    def forward(self, x, pool):
        x = self.ReLU(self.deconv1(x))
        x = self.ReLU(self.deconv2(x))
        x = self.deconv3(x)

        return self.ReLU(x)