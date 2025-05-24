import cv2
import matplotlib.pyplot as plt
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.tensorboard import TensorBoard
from Renderer.model import FCN,FCN_256
from Renderer.stroke_gen import *
import torch.optim as optim


def load_weights():
    pretrained_dict = torch.load('./checkpoints/renderer_model/render_3/renderer_499999.pth')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

net = FCN(input_dim=6)
if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    load_weights()
    # width = np.arange(0,4)/128
    for ll in range(2):
        # f = [0.7538343,0.50779283,0.80058306 ,0.7678529,  0.77189946, 0.46941024]
        #f = [0.13354631,0.93013721 ,0.85608183, 0.63978497, 0.03648514, 0.93085431,
             # 0.10250057, 0.09891955]
        f = np.random.uniform(0, 1, 2)  # 前六个是QBC三个点的坐标。最后一个是线的宽度
        f = np.tile(f,(3))
        # f[-1] = ll%5/5
        # f[-2:] = np.random.uniform(0, 15 / 128, 2)
        # f[-2] = 5/128
        # f[-1] = f[-2]

        # f[-2] = np.random.choice(width)
        # f[-2] = 0/128
        print(f)
        ground_truth = draw_circle_2(f,width=128)#[B-1,S-0]
        ground_truth_line = draw_line(f,width=128)#[B-1,S-0]
        pre_sketch = net(torch.tensor(np.array([f]),dtype=torch.float))#[B-1,S趋向于0，但不完全等于0]
        plt.subplot(1,3,1)
        plt.imshow(ground_truth_line,cmap='gray')
        # plt.show()
        plt.subplot(1,3,2)
        plt.imshow(ground_truth,cmap='gray')
        # plt.show()
        plt.subplot(1,3,3)
        pre_sketch_show = np.array(pre_sketch[0,:,:].detach())
        plt.imshow(pre_sketch_show,cmap='gray')
        plt.show()
        lgf = 1
        # exit()



