import torch
from utils.var_args import args
from Renderer.stroke_gen import draw_circle_2
import numpy as np
from Renderer.model import FCN
from Renderer.class_model import Class_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stork_num = 1#每次动作中笔画的数目
color_expand = torch.ones([1, 3], dtype=torch.float).to(device)
Decoder = FCN(input_dim=args.act_dim).to(device)#get [B:1,]
Decoder.load_state_dict(torch.load('./checkpoints/renderer_model/render_3/renderer_499999.pth'))
color_class = Class_model().to(device)
color_class.load_state_dict(torch.load('./checkpoints/class_model/class_1_20000.pth'))

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

def FreeDecode_BC(x,canvas):
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
    color_stroke_b1s0 = 1 - stroke * color_x.view(-1,1,1,1)#[B:0,S:1] -> [B:1,S:0]
    color_stroke_b1s0 = color_stroke_b1s0.permute(0, 3, 1, 2)
    color_stroke = stroke * (color_x*color_expand_).view(-1, 1, 1, 3) # [b*stork_num,width,width,1]*[b*stork_num,1,1,1] -> [b*stork_num,width,width,1]
    stroke = stroke.permute(0, 3, 1, 2)#[b*5,width,width,1] -> [b*5,1,width,width]
    color_stroke = color_stroke.permute(0, 3, 1, 2)#[b*5,width,width,1] -> [b*stork_num,1,width,width]
    stroke = stroke.view(-1, stork_num, 1, args.width,args.width)#[b*stork_num,1,width,width] -> [b,stork_num,1,width,width]
    color_stroke = color_stroke.view(-1, stork_num, 3, args.width, args.width)#[b*stork_num,1,width,width] -> [b,stork_num,1,width,width]
    for i in range(stork_num):
        canvas = canvas * (1-color_stroke[:,i]) #(1 - stroke[:, i]) + color_stroke[:, i]
    return canvas,color_stroke_b1s0

def decode_bc(x,canvas):
    x = x.view(-1, args.act_dim + 1)#[1,13]
    stroke = 1 - Decoder(x[:, :args.act_dim])#[1,128,128] [B:1,S:0] -> [B:0,S:1]
    stroke = stroke.view(-1, args.width, args.width, 1)#[1,128,128] -> [1,128,128,1]
    color_expand_ = color_expand.expand(x.shape[0], 3)
    color_stroke_b1s0 = 1 - stroke * color_class(x[:, -1:]).view(-1,1,1,1)
    color_stroke_b1s0 = color_stroke_b1s0.permute(0, 3, 1, 2)

    color_stroke = stroke * (color_class(x[:, -1:])*color_expand_).view(-1, 1, 1, 3) # [stork_num,128,128,1]*[stork_num,1,1,1] -> [stork_num,128,128,1]
    stroke = stroke.permute(0, 3, 1, 2)#[5,128,128,1] -> [5,1,128,128]
    color_stroke = color_stroke.permute(0, 3, 1, 2)#[5,128,128,1] -> [5,1,128,128]
    stroke = stroke.view(-1, stork_num, 1, args.width, args.width)#[5,1,128,128] -> [1,5,1,128,128]
    color_stroke = color_stroke.view(-1, stork_num, 3, args.width, args.width)#[5b,3,128,128] -> [b,5,3,128,128]
    for i in range(stork_num):
        # canvas = canvas * (1-stroke[:, i]) + color_stroke[:,i]#(1 - stroke[:, i]) + color_stroke[:, i]
        canvas = canvas * (1-color_stroke[:,i]) #(1 - stroke[:, i]) + color_stroke[:, i]
        # canvas = canvas * (1-stroke[:, i]) #+ color_stroke[:,i]#(1 - stroke[:, i]) + color_stroke[:, i]
    return canvas,color_stroke_b1s0

def decode_act_2_stroke(x):
    x = x.view(-1, args.act_dim + 1)#[1,13]
    stroke = 1 - Decoder(x[:, :args.act_dim])#[1,128,128] [B:1,S:0] -> [B:0,S:1]
    stroke = stroke.view(-1, args.width, args.width, 1)#[1,128,128] -> [1,128,128,1]
    color_stroke_b1s0 = 1 - stroke * color_class(x[:, -1:]).view(-1,1,1,1)
    color_stroke_b1s0 = color_stroke_b1s0.permute(0, 3, 1, 2)
    return color_stroke_b1s0