import torch
import torch.nn as nn
from torchvision import models

inp = torch.randn([2,3, 224, 224])
#print(inp.shape)


conv_block = nn.Sequential(nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
                           nn.BatchNorm2d(64), nn.ReLU(inplace = True), nn.MaxPool2d(kernel_size= 2, stride = 2))

ink = conv_block(inp)
#print(ink.shape)

class Basicblock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.ReLU = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ReLU(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.ReLU(out)

        return out


ewq = torch.randn((2, 64, 56, 56))
#print(Basicblock(64, 64)(ewq).shape)

def _make_layer(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride = stride, bias = False),
            nn.BatchNorm2d(planes),
        )
    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)

layers = [3, 4, 6, 3]
layer1 = _make_layer(Basicblock, 64, 64, layers[0], stride = 1 )
layer2 = _make_layer(Basicblock, 64, 128, layers[1], stride = 1)
layer3 = _make_layer(Basicblock, 128, 256, layers[2], stride = 1)
layer4 = _make_layer(Basicblock, 256, 512, layers[3], stride = 1)

#print(list(models.resnet34().children())[4])
#print(layer1)

t = torch.randn((2, 64, 56, 56))
o = nn.Conv2d(64, 128, 3, 2, 1)(t)
t_d = nn.Conv2d(64, 128, 1, 2, 0)(t)
print(t_d.shape, o.shape)

print((o+t_d).shape)
