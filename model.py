import torch
import torch.nn as nn
import torch.nn.functional as F

# model_paths = {'alexnet' : './model/alexnet.pt', 'vgg16':'./model/vgg16.pt'}

class AlexNet(nn.Module):
    def __init__(self, n_classes : int = 10):
        super(AlexNet, self).__init__()

        def crm2d(in_channels, out_channels, kernel_size, stride, padding, pool = True):
            layers = []
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding)]
            layers += [nn.ReLU(inplace = True)]
            if pool:
                layers += [nn.MaxPool2d(kernel_size = 3, stride = 2)]

            crm = nn.Sequential(*layers)

            return crm

        def dlr(n_classes):
            layers = []
            size_list = [(6 * 6* 256, 4096), (4096, 4096),(4096, n_classes)]
            for t in size_list[:-1]:
                layers +=[nn.Dropout(), nn.Linear(*t), nn.ReLU(inplace = True)]
            layers +=[nn.Linear(*size_list[-1])]
            dlr = nn.Sequential(*layers)

            return dlr

        self.arcs = [(3, 64, 11, 4, 2, True), (64, 192, 5, 1, 2, True), (192, 384, 3, 1, 1, False),
                     (384, 256, 3, 1, 1, False), (256, 256, 3, 1, 1, True)]
        self.layers = nn.ModuleDict()
        for i in range(len(self.arcs)):
            i_c, o_c, k, s, p, b = self.arcs[i]
            self.layers[f'conv{i}'] = crm2d(i_c, o_c, k, s, p, b)
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = dlr(n_classes)

    def forward(self, x):
        for i in range(len(self.arcs)):
            x = self.layers[f'conv{i}'](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGG(nn.Module):
    def __init__(self, n_classes : int, in_channels : int, mode):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.arc_dict = {'vgg11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                         'vgg16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
                                   'M', 512, 512, 512, 'M', 512, 512, 512,'M'],
                         'vgg19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
                                   512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}
        self.conv_layers = self.make_conv_layers(mode)
        self.classifier = self.make_classifier(n_classes)

    def make_conv_layers(self, mode):
        layers = []
        in_c = self.in_channels
        for c in self.arc_dict[mode]:
            if type(c) == int:
                layers += [nn.Conv2d(in_channels=in_c, out_channels=c, kernel_size=3, stride = 1, padding=1),
                           nn.BatchNorm2d(c),
                           nn.ReLU(inplace = True)]
                in_c = c
            elif c =='M':
                layers += [nn.MaxPool2d(2,2)]
            else:
                raise NotImplementedError

        return nn.Sequential(*layers)

    def make_classifier(self, n_classes):
        layers = []
        size_list = [(7 * 7 * 512, 4096), (4096, 4096), (4096, n_classes)]
        for t in size_list[:-1]:
            layers +=[nn.Dropout(), nn.Linear(*t), nn.ReLU(inplace=True)]
        layers+=[nn.Linear(*size_list[-1])]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        # x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

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
        x = self.relu(self.bn2(self.conv2(x)))

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
        x = self.relu(self.bn3(self.conv3(x)))

        if self.identity is not None:
            x_clone = self.identity(x_clone)

        x += x_clone
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, n_classes : int, in_channels : int, mode):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = 64
        self.arc_dict = {'resnet18' : (BasicBlock, [2, 2, 2, 2]),
                         'resnet34': (BasicBlock, [3, 4, 6, 3]),
                         'resnet50': (BottleNeckBlock, [3, 4, 6, 3]),
                         'resnet101' : (BottleNeckBlock, [3, 4, 23, 3]),
                         'resnet152' : (BottleNeckBlock, [3, 8, 36, 3])}
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.block_type = self.arc_dict[mode][0]
        self.num_blocks = self.arc_dict[mode][1]

        self.layer1 = self._make_layer(self.block_type, self.num_blocks[0], sub_channels = 64, stride = 1)
        self.layer2 = self._make_layer(self.block_type, self.num_blocks[1], sub_channels = 128, stride = 2)
        self.layer3 = self._make_layer(self.block_type, self.num_blocks[2], sub_channels = 256, stride = 2)
        self.layer4 = self._make_layer(self.block_type, self.num_blocks[3], sub_channels = 512, stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block_type.expansion, n_classes)

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






