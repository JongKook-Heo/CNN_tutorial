import torch
import torch.nn as nn
import torch.nn.functional as F

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