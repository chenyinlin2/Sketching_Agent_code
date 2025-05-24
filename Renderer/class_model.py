import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.optim as optim

class Class_model(nn.Module):
    def __init__(self):
        super(Class_model, self).__init__()
        self.fc1 = (nn.Linear(1, 30))
        self.fc2 = (nn.Linear(30, 30))
        self.fc3 = (nn.Linear(30, 1))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
def save_model(step):
    if use_cuda:
        Net.cpu()
    torch.save(Net.state_dict(), "../checkpoints/class_model/renderer_1_{}.pth".format(step))#renderer_256_2.pth绘制细笔画
    if use_cuda:
        Net.cuda()
    print('model of {} is saved'.format(step))
use_cuda = torch.cuda.is_available()
device = torch.device('cuda')
Net = Class_model().to(device)

if __name__ == '__main__':
    train_time = 20000

    criterion = nn.MSELoss()
    optimizer = optim.Adam(Net.parameters(), lr=1e-3)

    batch_size = 96
    step = 0
    while step < train_time:
        step += 1
        input = np.random.uniform(0,1,batch_size)[:,np.newaxis]
        # input = input
        gt = np.where(input > 0.5,1.,0.)

        input = torch.tensor(input,dtype=torch.float).to(device)
        gt = torch.tensor(gt,dtype=torch.float).to(device)

        target = Net(input)
        optimizer.zero_grad()
        loss = criterion(target, gt)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(step, loss.item())
        if step % 2000 == 0:
            Net.eval()
            input = np.random.uniform(0, 1, batch_size)[:,np.newaxis]
            gt = np.where(input < 0.5, 1., 0.)#
            input = torch.tensor(input, dtype=torch.float).to(device)
            gt = torch.tensor(gt, dtype=torch.float).to(device)
            target = Net(input)
            loss = criterion(gt, target)
            print('=======test=======')
            print('真值',gt[:])
            print('预期值',target[:])
            print('=======test=======')

        if step % 10000 == 0:
            save_model(step)





