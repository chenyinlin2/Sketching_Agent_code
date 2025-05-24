import random

import torch
from dataloader.QuickDraw_clean.var_shape_quickdraw import getQuickDrawTrain
import numpy as np
from utils.var_args import args
if __name__ == '__main__':
    train_loader = getQuickDrawTrain(batch_size=1,var_thick_flag=True)
    for i, sample_batch in enumerate(train_loader):
        del_white_num = 0
        for sub_idx in range(len(sample_batch)):  # 删掉其中的空白图像。即全是白色的图像
            sub_stroke_batch = sample_batch[sub_idx - del_white_num]
            all_white_flag = np.all(sub_stroke_batch == 255)  # 判断是否全部元素是255
            if all_white_flag:
                del sample_batch[sub_idx - del_white_num]
                del_white_num += 1
        sample_batch = torch.tensor(np.array(sample_batch), dtype=torch.float)
        gt = sample_batch.to(torch.uint8)  # 将gt的数据类型由int64 -> uint8
        cur_env_batch = gt.shape[0]  # env_batch被切割成了cur_env_batch
        gt = gt.expand(cur_env_batch, 3, args.width, args.width)
        print(gt.shape)
        # print(gt.nonzero())
        # gt_one = gt[:,0,:,:]
        random_start = torch.empty(cur_env_batch,2)
        for i in range(cur_env_batch):
            gt_one = gt[i,0,...]
            # gt_one = torch.ones(128,128)*255
            gt_zero_idx = (255-gt_one).nonzero()
            if len(gt_zero_idx) != 0:
                gt_zero_idx_idx = random.sample(range(gt_zero_idx.shape[0]),1)[0]
                random_start[i,:] = gt_zero_idx[gt_zero_idx_idx,:]
            else:
                random_start[i,:] = torch.zeros(1,2)

        break
# x = torch.tensor([1,2,3,4,5],dtype=torch.float,requires_grad=True)
    # y = 2*x + 1
    #
    # for yi in torch.flip(y,dims=[0]):
    #     print(yi)
    #     yi.backward(retain_graph=True)
    #     print(x.grad)

    # a = torch.ones(3, 1, 2)
    # print(a)
    # b = torch.ones(3, 2, 2)
    # print(b)
    # c = torch.matmul(a, b)
    # print(c.shape)
    # print(c)

    # a = torch.zeros(10, 2)
    # a.requires_grad = True
    # b = a.expand(128, 128, 10, 2)
    # b = b.permute(2, 3, 0, 1)
    # # b.requires_grad=True
    # print(b.requires_grad)
    # b_1 = torch.zeros(10,2,1,128)
    # b_1[:,:,0,0] = 1
    # # b_1.requires_grad = True
    # b_2 = torch.zeros(10, 2, 128, 1)
    # b_2[:, :, 0, 0] =  1
    # # b_2.requires_grad = True
    #
    # c = torch.matmul(torch.matmul(b_1, b), b_2).squeeze()
    # c.backward(torch.ones(10,2))
    # print(a.grad)
    # print(c)
