import torch
import torch.nn as nn
import torch.nn.functional as F

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