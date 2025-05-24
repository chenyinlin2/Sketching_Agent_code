import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 确定使用哪一块gpu
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.var_args import args
from DRL.actor import ResNet
from Renderer.stroke_gen import draw_circle_2
from dataloader.QuickDraw_clean.qucikdraw_loader import getQuickDrawVal
from Renderer.class_model import Class_model
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stork_num = 1#每次动作中笔画的数目
color_expand = torch.ones([1, 3], dtype=torch.float).to(device)

def FreeDecode(x,canvas): # b * (6 + 1) [1,1,width,width]
    x = x.view(-1, args.act_dim + 1)  # [b*stork_num,7]
    color_x = torch.where(x[:,-1:]>0.5,1.,0.)
    x_copy = x.detach().cpu().numpy()
    stroke = []
    color_expand_ = color_expand.expand(x.shape[0], 3)
    for i in range(x_copy.shape[0]):
        sub_storke = 1-draw_circle_2(x_copy[i,:args.act_dim],args.width)#[B:1,S:0] -> [B:0,S:1]
        # stroke = np.stack([storke,sub_storke],axis=0) #[b*5,width,width]
        stroke.append(sub_storke)
    stroke = torch.tensor(np.array(stroke)).to(canvas.device)
    stroke = stroke.view(-1,args.width, args.width, 1)#[b*stork_num,width,width] -> [b*stork_num,width,width,1]
    color_stroke = stroke * (color_x*color_expand_).view(-1, 1, 1, 3) # [b*stork_num,width,width,1]*[b*stork_num,1,1,1] -> [b*stork_num,width,width,1]
    stroke = stroke.permute(0, 3, 1, 2)#[b*5,width,width,1] -> [b*5,1,width,width]
    color_stroke = color_stroke.permute(0, 3, 1, 2)#[b*5,width,width,1] -> [b*stork_num,1,width,width]
    stroke = stroke.view(-1, stork_num, 1, args.width,args.width)#[b*stork_num,1,width,width] -> [b,stork_num,1,width,width]
    color_stroke = color_stroke.view(-1, stork_num, 3, args.width, args.width)#[b*stork_num,1,width,width] -> [b,stork_num,1,width,width]

    for i in range(stork_num):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    # return canvas,(stroke*(1-color_stroke))[0]#[b,1,width,width]
    # return canvas,(abs(stroke-color_stroke))[0]#[b,1,width,width]
    return canvas,stroke[0]#[b,1,width,width]
'''采用canvas*（1-stroke）+color_stroke的方式，代表1是绘制白色，0是绘制黑色。即能够绘制两种颜色。那么采用的color_class模型就应该在'''
'''采用canvas*(1-color_stroke)的方式，这时0和1分别表示画笔是否落下，如果输出，代表落下，则绘制出来黑色，如果不落下，则不会绘制'''
from Renderer.model import FCN
Decoder = FCN(input_dim=args.act_dim).to(device)#get [B:1,]
Decoder.load_state_dict(torch.load('./checkpoints/renderer_model/ori_render/renderer_499999.pth'))
color_class = Class_model().to(device)
color_class.load_state_dict(torch.load('./checkpoints/class_model/train_time_2/class_20000.pth'))

def decode(x, canvas): # b * (8)
    x = x.view(-1, args.act_dim + 1)#[1,13]
    stroke = 1 - Decoder(x[:, :args.act_dim])#[1,128,128] [B:1,S:0] -> [B:0,S:1]
    stroke = stroke.view(-1, args.width, args.width, 1)#[1,128,128] -> [1,128,128,1]
    color_expand_ = color_expand.expand(x.shape[0], 3)
    color_stroke = stroke * (color_class(x[:, -1:])*color_expand_).view(-1, 1, 1, 3) # [stork_num,128,128,1]*[stork_num,1,1,1] -> [stork_num,128,128,1]
    stroke = stroke.permute(0, 3, 1, 2)#[5,128,128,1] -> [5,1,128,128]
    color_stroke = color_stroke.permute(0, 3, 1, 2)#[5,128,128,1] -> [5,1,128,128]
    stroke = stroke.view(-1, stork_num, 1, args.width, args.width)#[5,1,128,128] -> [1,5,1,128,128]
    color_stroke = color_stroke.view(-1, stork_num, 3, args.width, args.width)#[5b,3,128,128] -> [b,5,3,128,128]
    for i in range(stork_num):
        canvas = canvas * (1-stroke[:, i]) + color_stroke[:,i]#(1 - stroke[:, i]) + color_stroke[:, i]
    return canvas,(stroke*torch.ones_like(1-color_stroke))[0]
'''采用canvas*（1-stroke）+color_stroke的方式，代表1是绘制白色，0是绘制黑色。即能够绘制两种颜色。那么采用的color_class模型就应该在'''
'''采用canvas*(1-color_stroke)的方式，这时0和1分别表示画笔是否落下，如果输出，代表落下，则绘制出来黑色，如果不落下，则不会绘制'''

def decode_one(x, canvas): # b * (8)
    x = x.view(-1, args.act_dim + 1)#[1,13]
    stroke = 1 - Decoder(x[:, :args.act_dim])#[1,128,128] [B:1,S:0] -> [B:0,S:1]
    stroke = stroke.view(-1, args.width, args.width, 1)#[1,128,128] -> [1,128,128,1]
    color_expand_ = color_expand.expand(x.shape[0], 3)
    color_stroke = stroke * (color_class(x[:, -1:])*color_expand_).view(-1, 1, 1, 3) # [stork_num,128,128,1]*[stork_num,1,1,1] -> [stork_num,128,128,1]
    stroke = stroke.permute(0, 3, 1, 2)#[5,128,128,1] -> [5,1,128,128]
    color_stroke = color_stroke.permute(0, 3, 1, 2)#[5,128,128,1] -> [5,1,128,128]
    stroke = stroke.view(-1, stork_num, 1, args.width, args.width)#[5,1,128,128] -> [1,5,1,128,128]
    color_stroke = color_stroke.view(-1, stork_num, 3, args.width, args.width)#[5b,3,128,128] -> [b,5,3,128,128]
    for i in range(stork_num):
        # canvas = canvas * (1-stroke[:, i]) + color_stroke[:,i]#(1 - stroke[:, i]) + color_stroke[:, i]
        canvas = canvas * (1-color_stroke[:,i]) #(1 - stroke[:, i]) + color_stroke[:, i]
        # canvas = canvas * (1-stroke[:, i]) #+ color_stroke[:,i]#(1 - stroke[:, i]) + color_stroke[:, i]
    return canvas,stroke[0]#(stroke*color_stroke)[0]
def FreeDecode_one(x,canvas): # b * (6 + 1) [1,1,width,width]
    x = x.view(-1, args.act_dim + 1)  # [b*stork_num,7]
    color_x = torch.where(x[:,-1:]>0.5,1.,0.)
    x_copy = x.detach().cpu().numpy()
    stroke = []
    color_expand_ = color_expand.expand(x.shape[0], 3)
    for i in range(x_copy.shape[0]):
        sub_storke = 1-draw_circle_2(x_copy[i,:args.act_dim],args.width)#[B:1,S:0] -> [B:0,S:1]
        # stroke = np.stack([storke,sub_storke],axis=0) #[b*5,width,width]
        stroke.append(sub_storke)
    stroke = torch.tensor(np.array(stroke)).to(canvas.device)
    stroke = stroke.view(-1,args.width, args.width, 1)#[b*stork_num,width,width] -> [b*stork_num,width,width,1]
    color_stroke = stroke * (color_x*color_expand_).view(-1, 1, 1, 3) # [b*stork_num,width,width,1]*[b*stork_num,1,1,1] -> [b*stork_num,width,width,1]
    stroke = stroke.permute(0, 3, 1, 2)#[b*5,width,width,1] -> [b*5,1,width,width]
    color_stroke = color_stroke.permute(0, 3, 1, 2)#[b*5,width,width,1] -> [b*stork_num,1,width,width]
    stroke = stroke.view(-1, stork_num, 1, args.width,args.width)#[b*stork_num,1,width,width] -> [b,stork_num,1,width,width]
    color_stroke = color_stroke.view(-1, stork_num, 3, args.width, args.width)#[b*stork_num,1,width,width] -> [b,stork_num,1,width,width]
    for i in range(stork_num):
        canvas = canvas * (1-color_stroke[:,i]) #(1 - stroke[:, i]) + color_stroke[:, i]
    return canvas,stroke[0]
def save_img(res, imgid, divide=False):
    output = res.detach().cpu().numpy() # d * d, 3, width, width
    output = np.transpose(output, (0, 2, 3, 1))
    output = (output[0] * 255).astype('uint8')
    cv2.imwrite('output/generated' + str(imgid) + '.png', output)
if __name__ == '__main__':
    actor = ResNet(args.obs_dim+2+2, 18, args.act_dim -2 +1)  # action_bundle = 5, 65 = 5 * 13
    actor.load_state_dict(torch.load(args.actor_check))
    actor = actor.to(device).eval()
    canvas = torch.ones([1, 3, args.width, args.width]).to(device)
    batch_size = 1
    # np.random.seed(100)
    # random.seed(100)
    val_dataloader = getQuickDrawVal(batch_size=batch_size)
    var_thick_flag = args.var_thick_flag#是否使用不同宽度的画笔绘制不同大小的图片的标志位

    for i,sample_batch in enumerate(val_dataloader):

        # plt.imshow(sample_batch[0, 0, :, :], cmap='gray')
        # plt.show()
        # if i > 50:
        #     exit()
        # continue
        img = sample_batch.detach().to(torch.float32)
        img = img.to(device)
        gt = img.expand(batch_size, 3, args.width, args.width)
        break

    show_canvas = gt.permute(0,2,3,1)[0,:,:,:].cpu().numpy()
    plt.imshow(show_canvas,cmap ='gray')
    plt.show()
    # img_zeros = torch.ones_like(img,dtype=torch.float).to(device)
    T = torch.ones([1, 1, args.width, args.width], dtype=torch.float32).to(device)
    coord = torch.zeros([1, 2, args.width, args.width])
    for i in range(args.width):
        for j in range(args.width):
            coord[0, 0, i, j] = i / (args.width - 1.)
            coord[0, 1, i, j] = j / (args.width - 1.)
    coord = coord.to(device)

    stroke_start = torch.rand(1, 2, dtype=torch.float).to(device)  # 笔画的初始位置

    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    with torch.no_grad():
        for i in range(args.max_step):
            stepnum = T * i / args.max_step
            stroke_start_input = stroke_start.expand(args.width, args.width, 1, 2).permute(2, 3, 0, 1)  # [batch,2] -> [batch,2,128,128]

            actions = actor(torch.cat([stroke_start_input,canvas, gt/255, stepnum, coord], 1))
            actions = torch.cat([stroke_start,actions],dim=1)
            stroke_start = actions[:,4:6]

            stroke_end_point = actions[:, 4:6].view(batch_size, 2, 1).expand(batch_size, 2, 3)
            stroke_end_point = stroke_end_point.permute(0, 2, 1).reshape(batch_size, -1)
            stroke_end_point_canvas = 1 - Decoder(stroke_end_point)  # [B:1,S:0] -> [B:0,S:1]
            stroke_end_point_canvas = stroke_end_point_canvas.view(batch_size, args.width, args.width, 1).expand(
                batch_size, args.width, args.width, 3).permute(0, 3,
                                                                     1, 2)
            stroke_end_point_pixel = 300 * ((0.5 - gt/255) * stroke_end_point_canvas).view(batch_size, -1).mean(dim=1,
                                                                                                                keepdims=True)
            print('reward of step ',i+1,' is ', stroke_end_point_pixel)
            print('actions',actions.detach().cpu().numpy())
            actions[:,-1:] = color_class(actions[:,-1:])
            print('actions',actions.detach().cpu().numpy())
            # canvas,stroke = decode(actions,canvas)
            canvas,stroke = FreeDecode_one(actions,canvas)
            color_rgb = torch.tensor([[255,64,64]]).to(device)/255
            color_stroke = stroke.expand(1,3,args.width,args.width)#[1,3,128,128]
            gt_with_stroke = gt*(1-stroke)/255 + color_stroke*color_rgb.view(1,3,1,1)

            show_canvas = canvas.permute(0, 2, 3, 1)[0].cpu().numpy()
            show_strokes = gt_with_stroke.permute(0,2,3,1)[0].cpu().numpy()
            show_stroke_end_point_canvas = ((1 - gt/255) * stroke_end_point_canvas).permute(0,2,3,1)[0].cpu().numpy()

            ax1 = fig1.add_subplot(6,7,i+1)
            ax1.axis('off')
            ax1.imshow(show_canvas)
            ax2 = fig2.add_subplot(6,7,i+1)
            ax2.axis('off')
            ax2.imshow(show_strokes)
            ax3 = fig3.add_subplot(6,7,i+1)
            ax3.axis('off')
            ax3.imshow(show_stroke_end_point_canvas)
    plt.show()
    plt.title('final canvas')
    plt.imshow(show_canvas)
    plt.show()
    # show_canvas = canvas.permute(0,2,3,1)[0].cpu().numpy()
    # plt.imshow(show_canvas)
    # plt.show()

    print('okk')