import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, sub_channels, identity = None, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, sub_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(sub_channels)
        self.conv2 = nn.Conv2d(sub_channels, sub_channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(sub_channels)
        self.relu = nn.ReLU()
        self.identity = identity
        self.stride = stride

    def forward(self, x):
        x_clone = x.clone()

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.identity is not None:
            x_clone = self.identity(x_clone)

        x += x_clone
        x = self.relu(x)
        return x

class BottleNeckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, sub_channels, identity = None, stride = 1):
        super(BottleNeckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, sub_channels, kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(sub_channels)
        self.conv2 = nn.Conv2d(sub_channels, sub_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn2 = nn.BatchNorm2d(sub_channels)
        self.conv3 = nn.Conv2d(sub_channels, sub_channels * self.expansion, kernel_size = 1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(sub_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity = identity
        self.stride = stride

    def forward(self, x):
        x_clone = x.clone()

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.identity is not None:
            x_clone = self.identity(x_clone)

        x += x_clone
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, n_classes : int, in_channels : int, mode, k = 1):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = 64
        self.arc_dict = {'resnet18' : (BasicBlock, [2, 2, 2, 2], k),
                         'resnet34': (BasicBlock, [3, 4, 6, 3], k),
                         'resnet50': (BottleNeckBlock, [3, 4, 6, 3], k),
                         'resnet101' : (BottleNeckBlock, [3, 4, 23, 3], k),
                         'resnet152' : (BottleNeckBlock, [3, 8, 36, 3], k)}
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.block_type = self.arc_dict[mode][0]
        self.num_blocks = self.arc_dict[mode][1]
        self.w_factor = self.arc_dict[mode][2]

        self.layer1 = self._make_layer(self.block_type, self.num_blocks[0], sub_channels = 64 * self.w_factor, stride = 1)
        self.layer2 = self._make_layer(self.block_type, self.num_blocks[1], sub_channels = 128 * self.w_factor, stride = 2)
        self.layer3 = self._make_layer(self.block_type, self.num_blocks[2], sub_channels = 256 * self.w_factor, stride = 2)
        self.layer4 = self._make_layer(self.block_type, self.num_blocks[3], sub_channels = 512 * self.w_factor, stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block_type.expansion * self.w_factor, n_classes)

    def _make_layer(self, block, num_blocks, sub_channels, stride):
        identity = None
        layers = []

        if stride != 1 or self.inter_channels != sub_channels * block.expansion:
            identity = nn.Sequential(
                nn.Conv2d(self.inter_channels, sub_channels * block.expansion, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(sub_channels * block.expansion))
        layers.append(block(self.inter_channels, sub_channels, identity, stride))

        self.inter_channels = sub_channels * block.expansion
        for i in range(num_blocks - 1):
            layers.append(block(self.inter_channels, sub_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x








