import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

class FCN(nn.Module):
    def __init__(self,input_dim=8):
        super(FCN, self).__init__()
        self.fc1 = (nn.Linear(input_dim, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv6 = (nn.Conv2d(8, 4, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):#[64,10]
        x = F.relu(self.fc1(x))#[64,10] -> [64,512]
        x = F.relu(self.fc2(x))#[64,512] -> [64,1024]
        x = F.relu(self.fc3(x))#[64,1024] -> [64,2048]
        x = F.relu(self.fc4(x))#[64,2048] -> [64,4096]
        x = x.view(-1, 16, 16, 16)#[64,4096] -> [64,16,16,16]
        x = F.relu(self.conv1(x))#[64,16,16,16] -> [64,32,16,16]
        x = self.pixel_shuffle(self.conv2(x))#[64,32,16,16] -> [64,8,32,32]
        x = F.relu(self.conv3(x))#[64,8,32,32] -> [64,16,32,32]
        x = self.pixel_shuffle(self.conv4(x))#[64,16,32,32] -> [64,4,64,64]
        x = F.relu(self.conv5(x))#[64,4,64,64] -> [64,8,64,64]
        x = self.pixel_shuffle(self.conv6(x))#[64,8,64,64] -> [64,1,128,128]
        x = torch.sigmoid(x)#[64,1,128,128]
        return 1 - x.view(-1, 128, 128)

class FCN_256(nn.Module):
    def __init__(self):
        super(FCN_256, self).__init__()
        self.fc1 = (nn.Linear(6, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv6 = (nn.Conv2d(8, 8, 3, 1, 1))
        self.conv7 = (nn.Conv2d(2, 4, 3, 1, 1))
        self.conv8 = (nn.Conv2d(4, 4, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):#[64,10]
        x = F.relu(self.fc1(x))#[64,10] -> [64,512]
        x = F.relu(self.fc2(x))#[64,512] -> [64,1024]
        x = F.relu(self.fc3(x))#[64,1024] -> [64,2048]
        x = F.relu(self.fc4(x))#[64,2048] -> [64,4096]
        x = x.view(-1, 16, 16, 16)#[64,4096] -> [64,16,16,16]
        x = F.relu(self.conv1(x))#[64,16,16,16] -> [64,32,16,16]
        x = self.pixel_shuffle(self.conv2(x))#[64,32,16,16] -> [64,8,32,32]
        x = F.relu(self.conv3(x))#[64,8,32,32] -> [64,16,32,32]
        x = self.pixel_shuffle(self.conv4(x))#[64,16,32,32] -> [64,4,64,64]
        x = F.relu(self.conv5(x))#[64,4,64,64] -> [64,8,64,64]
        x = self.pixel_shuffle(self.conv6(x))#[64,8,64,64] -> [64,2,128,128]
        x = F.relu(self.conv7(x))#[64,2,128,128] -> [64,4,128,128]
        x = self.pixel_shuffle(self.conv8(x))#[64,4,128,128] -> [64,1,256,256]
        x = torch.sigmoid(x)#[64,1,128,128]
        return 1 - x.view(-1, 256, 256)
