import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.tensorboard import TensorBoard
from Renderer.model import FCN,FCN_256
from Renderer.stroke_gen import *
import torch.optim as optim
import os

def save_model(step):
    if use_cuda:
        net.cpu()
    torch.save(net.state_dict(), "./checkpoints/renderer_model/render256_1/renderer_{}.pth".format(step))#renderer_256_2.pth绘制细笔画
    if use_cuda:
        net.cuda()
    print('model of {} is saved'.format(step))
def load_weights():
    pretrained_dict = torch.load("./checkpoints/renderer_model/renderer_3.pth")
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

net = FCN_256()
if __name__ == '__main__':
    writer = TensorBoard("./render_logs/train256_log_1/")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=3e-6)
    batch_size = 64
    use_cuda = torch.cuda.is_available()
    # load_weights()
    step = 0
    # load_weights()
    while step < 500000:
        net.train()
        train_batch = []
        ground_truth = []
        for i in range(batch_size):
            f = np.random.uniform(0, 1, 6)#前六个是QBC三个点的坐标。最后一个是线的宽度

            # f[-1] = np.random.choice([0,1,2,3],1)/128
            # f[-2] = np.random.choice([0,1,2,3],1)/128
            # f[-4] = f[-3] * 1

            train_batch.append(f)
            ground_truth.append(draw_circle_for256(f,width=256))#[B:1,S:0]

        train_batch = torch.tensor(train_batch).float()
        ground_truth = torch.tensor(ground_truth).float()
        if use_cuda:
            net = net.cuda()
            train_batch = train_batch.cuda()
            ground_truth = ground_truth.cuda()
        gen = net(train_batch)
        optimizer.zero_grad()
        loss = criterion(gen, ground_truth)
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print(step, loss.item())
        if step < 200000:
            lr = 1e-4
        elif step < 400000:
            lr = 1e-5
        else:
            lr = 1e-6
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        writer.add_scalar("train/loss", loss.item(), step)
        if step % 5000 == 0:
            net.eval()
            gen = net(train_batch)
            loss = criterion(gen, ground_truth)
            writer.add_scalar("val/loss", loss.item(), step)
            for i in range(32):
                G = gen[i].cpu().data.numpy()
                GT = ground_truth[i].cpu().data.numpy()
                writer.add_image("train_{}/ground_truth.png".format(i), GT*255, step)
                writer.add_image("train_{}/gen.png".format(i), G*255, step)
        if (step+1) % 50000 == 0:
            save_model(step)
        step += 1
