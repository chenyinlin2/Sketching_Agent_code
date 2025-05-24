import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, SGD
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import torch.nn.utils.weight_norm as weightNorm
class TReLU(nn.Module):
    def __init__(self):
            super(TReLU, self).__init__()
            self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x

class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()

            self.conv0 = weightNorm(nn.Conv2d(6, 16, 5, 2, 2))
            self.conv1 = weightNorm(nn.Conv2d(16, 32, 5, 2, 2))
            self.conv2 = weightNorm(nn.Conv2d(32, 64, 5, 2, 2))
            self.conv3 = weightNorm(nn.Conv2d(64, 128, 5, 2, 2))
            self.conv4 = weightNorm(nn.Conv2d(128, 1, 5, 2, 2))
            self.relu0 = TReLU()
            self.relu1 = TReLU()
            self.relu2 = TReLU()
            self.relu3 = TReLU()

        def forward(self, x):#[batch,3,128,128]
            x = self.conv0(x)#[batch,3,128,128] -> [batch,16,64,64]
            x = self.relu0(x)#[batch,16,64,64] -> [batch,16,64,64]
            x = self.conv1(x)#[batch,16,64,64] -> [batch,32,32,32]
            x = self.relu1(x)#[batch,32,32,32] -> [batch,32,32,32]
            x = self.conv2(x)#[batch,32,32,32] -> [batch,64,16,16]
            x = self.relu2(x)#[batch,64,16,16] -> [batch,64,16,16]
            x = self.conv3(x)#[batch,64,16,16] -> [batch,128,8,8]
            x = self.relu3(x)#[batch,128,8,8] -> [batch,128,8,8]
            x = self.conv4(x)#[batch,128,8,8] -> [batch,1,4,4]
            x = F.avg_pool2d(x, 4)#[batch,1,1,1]
            x = x.view(-1, 1)#[batch,1]
            # x = F.tanh(x)
            return x