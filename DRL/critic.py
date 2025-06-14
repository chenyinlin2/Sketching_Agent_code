import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
from torch.autograd import Variable
import sys


def conv3x3(in_planes, out_planes, stride=1):
    return weightNorm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coord = torch.zeros([1, 2, 64, 64])
for i in range(64):
    for j in range(64):
        coord[0, 0, i, j] = i / 63.
        coord[0, 1, i, j] = j / 63.
        coord = coord.to(device)


class TReLU(nn.Module):
    def __init__(self):
        super(TReLU, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x


def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2, 2, 2, 2]),
        '34': (BasicBlock, [3, 4, 6, 3]),
        '50': (Bottleneck, [3, 4, 6, 3]),
        '101': (Bottleneck, [3, 4, 23, 3]),
        '152': (Bottleneck, [3, 8, 36, 3]),
    }

    return cf_dict[str(depth)]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                weightNorm(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True)),
            )
        self.relu_1 = TReLU()
        self.relu_2 = TReLU()

    def forward(self, x):
        out = self.relu_1(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu_2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = weightNorm(nn.Conv2d(in_planes, planes, kernel_size=1, bias=True))
        self.conv2 = weightNorm(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True))
        self.conv3 = weightNorm(nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=True))
        self.relu_1 = TReLU()
        self.relu_2 = TReLU()
        self.relu_3 = TReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                weightNorm(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True)),
            )

    def forward(self, x):
        out = self.relu_1(self.conv1(x))
        out = self.relu_2(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = self.relu_3(out)

        return out

class ResNet_wobn(nn.Module):
    def __init__(self, num_inputs, depth, num_outputs):
        super(ResNet_wobn, self).__init__()
        self.in_planes = 64

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(num_inputs, 64, 2)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_outputs)
        self.relu_1 = TReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu_1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# class ResNet_wobn(nn.Module):
#     def __init__(self, num_inputs, depth, num_outputs):  # 3 + 2, 18, 1
#         super(ResNet_wobn, self).__init__()
#         self.in_planes = 64
#
#         block, num_blocks = cfg(depth)
#         self.conv0 = conv3x3(num_inputs, 32, 2)  # 64
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)  # 32
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 16
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.conv4 = weightNorm(nn.Conv2d(512, 1, 1, 1, 0))
#         self.relu_1 = TReLU()
#         self.conv1 = weightNorm(nn.Conv2d(args.act_dim + 2, 64, 1, 1, 0))
#         self.conv2 = weightNorm(nn.Conv2d(64, 64, 1, 1, 0))
#         self.conv3 = weightNorm(nn.Conv2d(64, 32, 1, 1, 0))
#         self.relu_2 = TReLU()
#         self.relu_3 = TReLU()
#         self.relu_4 = TReLU()
#         self.fc = nn.Linear(512 * block.expansion, num_outputs)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#
#         return nn.Sequential(*layers)
#
#     def a2img(self, x):  # [96,8]
#         tmp = coord.expand(x.shape[0], 2, 64, 64)  # [96,2,64,64]
#         x = x.repeat(64, 64, 1, 1).permute(2, 3, 0, 1)  # [96,args.act_dim] -> [96,args.act_dim,64,64]
#         x = self.relu_2(self.conv1(torch.cat([x, tmp], 1)))#[96,args.act_dim+2,64,64] -> [96,64,64,64]
#         x = self.relu_3(self.conv2(x))  # [96,64,64,64]->  [96,64,64,64]
#         x = self.relu_4(self.conv3(x))  # [96,64,64,64]-> [96,32,64,64]
#         return x
#
#     def forward(self, input):
#         x, a = input  # 状态x:[96,5,128,128] | 动作a:[96,7]
#         a = self.a2img(a) #[96,8] -> [96,32,64,64]
#         x = self.relu_1(self.conv0(x))  # [96,5,128,128] -> [96,32,128,128]
#         x = torch.cat([x, a], 1)  # [96,9,128,128] -> [96,64,128,128]
#         x = self.layer1(x)  # [96,64,128,128] -> [96,64,32,32]
#         x = self.layer2(x)  # [96,64,32,32] -> [96,128,16,16]
#         x = self.layer3(x)  # [96,128,16,16] -> [96,256,8,8]
#         x = self.layer4(x)  # [96,256,8,8] -> [96,512,8,8]
#         x = F.avg_pool2d(x, 4)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
