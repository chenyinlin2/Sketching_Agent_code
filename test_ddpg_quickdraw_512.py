import os
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
    return canvas,(stroke*(1-color_stroke))[0]#[b,1,width,width]
'''采用canvas*（1-stroke）+color_stroke的方式，代表1是绘制白色，0是绘制黑色。即能够绘制两种颜色。那么采用的color_class模型就应该在'''
'''采用canvas*(1-color_stroke)的方式，这时0和1分别表示画笔是否落下，如果输出，代表落下，则绘制出来黑色，如果不落下，则不会绘制'''
from Renderer.model import FCN
Decoder = FCN(input_dim=args.act_dim).to(device)#get [B:1,]
Decoder.load_state_dict(torch.load('./checkpoints/renderer_model/render_3/renderer_499999.pth'))
color_class = Class_model().to(device)
color_class.load_state_dict(torch.load('./checkpoints/class_model/class_1_20000.pth'))
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
    return canvas
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
    return canvas,(stroke*color_stroke)[0]
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
    return canvas,(stroke*color_stroke)[0]
'''采用canvas*（1-stroke）+color_stroke的方式，代表1是绘制白色，0是绘制黑色。即能够绘制两种颜色。那么采用的color_class模型就应该在'''
'''采用canvas*(1-color_stroke)的方式，这时0和1分别表示画笔是否落下，如果输出，代表落下，则绘制出来黑色，如果不落下，则不会绘制'''
# os.system('mkdir output')
def save_img(res, imgid, divide=False):
    output = res.detach().cpu().numpy() # d * d, 3, width, width
    output = np.transpose(output, (0, 2, 3, 1))
    output = (output[0] * 255).astype('uint8')
    cv2.imwrite('output/generated' + str(imgid) + '.png', output)
def large2small(x):#[512,512,3]
    # (d * width, d * width) -> (d * d, width, width)
    x = x.reshape(args.divide, args.width, args.divide, args.width, 3)#[512,512,3] -> [4,128,4,128,3]
    x = np.transpose(x, (0, 2, 1, 3, 4))#[4,128,4,128,3] -> [4,4,128,128,3]
    x = x.reshape(args.divide**2, args.width, args.width, 3)#[4,4,128,128] -> [16,128,128,3]
    return x
def small2large(x):
    # (d * d, width, width) -> (d * width, d * width)
    x = x.reshape(args.divide, args.divide, args.width, args.width, -1)#[16,128,128,3] -> [4,4,128,128,3]
    x = np.transpose(x, (0, 2, 1, 3, 4))#[4,4,128,128,3] -> [4,128,4,128,3]
    x = x.reshape(args.divide * args.width, args.divide * args.width, -1)#[4,128,4,128,3] -> [512,512,3]
    return x
if __name__ == '__main__':
    actor = ResNet(9, 18, args.act_dim+1)  # action_bundle = 5, 65 = 5 * 13
    actor.load_state_dict(torch.load(args.good_actor_check))
    actor = actor.to(device).eval()
    canvas = torch.ones([1, 3, args.width, args.width]).to(device)
    batch_size = 1
    np.random.seed(100)
    random.seed(100)
    width = 512
    val_dataloader = getQuickDrawVal(batch_size=batch_size,img_size=width)
    args.divide = width//args.width
    '''随机抽取一张测试集图像。'''
    # for i,sample_batch in enumerate(val_dataloader):
    #     img = sample_batch.detach().to(torch.float32)#[batch,1,128,128]
    #     img = img.to(device)
    #     img = img.expand(batch_size, 3, args.width*args.divide, args.width*args.divide)#[batch,1,128,128] -> [batch,3,128,128]
    #     break
    # show_canvas = img.permute(0,2,3,1)[0,:,:,:].cpu().numpy()#[batch,3,128,128] -> [128,128,3]

    '''读取一张非常大的图片'''
    from utils.img_process import get_big_pad_img
    img_path = '/media/lgf/rl_draw_image/rl_draw_edge_render_change_error/sample_inputs/clean_line_drawings/sheriff.png'
    show_canvas = get_big_pad_img(img_path)#[width,width,3]
    real_width = show_canvas.shape[0]
    args.divide = real_width//args.width

    plt.imshow(show_canvas)
    plt.show()


    #[128,128,3] —> [512,512,3]
    # patch_img = cv2.resize(show_canvas, (args.width * args.divide,args.width * args.divide),interpolation=cv2.INTER_AREA)
    # plt.imshow(patch_img)
    # plt.show()
    patch_img = large2small(show_canvas)  # [512,512,3] -> [16,128,128,3]
    # 一张图片被放大到512*512后，再裁剪为128*128.进行显示。
    fig0 = plt.figure()
    for i in range(args.divide ** 2):
        patch_img_show = patch_img[i,:,:,:]
        ax0 = fig0.add_subplot(args.divide , args.divide , i + 1)
        plt.xticks([])
        plt.yticks([])
        ax0.imshow(patch_img_show)
    plt.show()

    patch_img = np.transpose(patch_img, (0, 3, 1, 2))  # [16,128,128,3] -> [16,3,128,128]
    patch_img = torch.tensor(patch_img).to(device).float() / 255.
    canvas = torch.ones_like(patch_img,dtype=torch.float)#[16,3,128,128]


    T = torch.ones([args.divide ** 2, 1, args.width, args.width], dtype=torch.float).to(device)
    coord = torch.zeros([1, 2, args.width, args.width],dtype=torch.float)
    for i in range(args.width):
        for j in range(args.width):
            coord[0, 0, i, j] = i / (args.width - 1.)
            coord[0, 1, i, j] = j / (args.width - 1.)
    coord = coord.to(device)
    coord = coord.expand(args.divide ** 2, 2, args.width, args.width)

    fig1 = plt.figure()

    with torch.no_grad():
        for i in range(args.max_step):
            stepnum = T * i / args.max_step
            actions = actor(torch.cat([canvas, patch_img, stepnum, coord], 1))
            # print(actions)
            canvas,stroke = FreeDecode(actions,canvas)
            # canvas,stroke = FreeDecode_one(actions,canvas)
            canvas_copy = np.transpose(canvas.detach().cpu().numpy(),(0,2,3,1))#[16,3,128,128] -> [16,128,128,3]
            show_output = small2large(canvas_copy)#[16,128,128,3] -> [512,512,3]
            ax1 = fig1.add_subplot(6,7,i+1)
            plt.xticks([])
            plt.yticks([])
            ax1.imshow(show_output)
        plt.show()
    plt.show()

    fig2 = plt.figure()
    for i in range(args.divide*args.divide):
        ax2 = fig2.add_subplot(args.divide , args.divide , i + 1)
        plt.xticks([])
        plt.yticks([])
        ax2.imshow(canvas_copy[i])
    plt.show()

    plt.imshow(show_output)
    plt.title('final canvas')
    plt.show()
    print('okk')

