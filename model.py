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

# class AlexNet_revised(nn.Module):
#     def __init__(self, n_classes : int = 10):
#         super(AlexNet_revised, self).__init__()
#
#         def crm2d(in_channels, out_channels, kernel_size, stride, padding, pool = True):
#             layers = []
#             layers += [nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding)]
#             layers += [nn.ReLU(inplace = True)]
#             if pool:
#                 layers += [nn.MaxPool2d(kernel_size = 3, stride = 2)]
#
#             crm = nn.Sequential(*layers)
#
#             return crm
#
#         def dlr(n_classes):
#             layers = []
#             size_list = [(6 * 6* 256, 4096), (4096, 4096),(4096, n_classes)]
#             for t in size_list[:-1]:
#                 layers +=[nn.Dropout(), nn.Linear(*t), nn.ReLU(inplace = True)]
#             layers +=[nn.Linear(*size_list[-1])]
#             dlr = nn.Sequential(*layers)
#
#             return dlr
#
#
#         self.f_map1 = crm2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding = 2)
#         self.f_map2 = crm2d(in_channels=64, out_channels=192, kernel_size=5, stride = 1, padding = 2)
#         self.f_map3 = crm2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding = 1, pool = False)
#         self.f_map4 = crm2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding = 1, pool = False)
#         self.f_map5 = crm2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding = 1, pool = True)
#         self.avgpool = nn.AdaptiveAvgPool2d((6,6))
#         self.classifier = dlr(n_classes)
#
#     def forward(self, x):
#         x = self.f_map1(x)
#         x = self.f_map2(x)
#         x = self.f_map3(x)
#         x = self.f_map4(x)
#         x = self.f_map5(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

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

class ResNet(nn.Module):
    def __init__(self, n_classes : int, in_channels : int, mode):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.arc_dict = {}

    def forward(self, x):
        pass






