import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                     groups=1, bias=False, dilation=1)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetBlock(nn.Module):
    def __init__(self, inplanes, planes, strides=1):
        super(ResNetBlock, self).__init__():
        # norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(input)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if (stride[0] > 1) or (input.shape.)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64

        for i in range(3):
            self.layer1 = ResNetBlock(self.inplanes, 64)

        self.layer2 = ResNetBlock(self.)
        for i in range(4):

    def forward(self, x):

