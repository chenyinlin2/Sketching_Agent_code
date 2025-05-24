import torch
from matplotlib import pyplot as plt
from DRL.wgan import *
from utils.var_args import args
from dataloader.QuickDraw_clean.qucikdraw_loader import getQuickDrawTrain,getQuickDrawVal
from dataloader.img_align_celeba.img_align_dataloader import getAlign_celceba_data
from Renderer.stroke_gen import draw_line,draw_circle_2
'''测试人脸图像在训练好的gan网络的计算情况'''
# if __name__ == '__main__':
#     load_gan(args.resume)
#     datapath = '/media/data/img_align_celeba'
#     train_loader, test_loader = getAlign_celceba_data(data_base_path=datapath, train_batch=64)
#     for i, sample_batch in enumerate(test_loader):
#         train_image = sample_batch.to(torch.uint8).to(device)
#         canvas = torch.ones_like(train_image)*255
#
#         gt = train_image.float() / 255
#         canvas = canvas.float() / 255
#
#         reward_feak = cal_reward(canvas,gt)
#         reward_real = cal_reward(gt,gt)
#         print('feak_r:',reward_feak)
#         print('real_r:',reward_real)
#         if i > 10:
#             break
'''测试线条图在训练好的gan网络的计算情况'''

if __name__ == '__main__':
    from Renderer.model import FCN
    from Renderer.class_model import Class_model
    Decoder = FCN(input_dim=args.act_dim).to(device)  # get [B:1,]
    Decoder.load_state_dict(torch.load('./checkpoints/renderer_model/render_5/renderer_499999.pth'))
    color_class = Class_model().to(device)
    color_class.load_state_dict(torch.load('./checkpoints/class_model/class_1_20000.pth'))
    stork_num = 1
    color_expand = torch.ones([1, 3], dtype=torch.float).to(device)


    def decode(x, canvas):  # b * (8)
        x = x.view(-1, args.act_dim + 1)  # [1,13]
        stroke = 1 - Decoder(x[:, :args.act_dim])  # [1,128,128] [B:1,S:0] -> [B:0,S:1]
        stroke = stroke.view(-1, args.width, args.width, 1)  # [1,128,128] -> [1,128,128,1]
        color_expand_ = color_expand.expand(x.shape[0], 3)
        color_stroke = stroke * (color_class(x[:, -1:]) * color_expand_).view(-1, 1, 1,
                                                                              3)  # [stork_num,128,128,1]*[stork_num,1,1,1] -> [stork_num,128,128,1]
        stroke = stroke.permute(0, 3, 1, 2)  # [5,128,128,1] -> [5,1,128,128]
        color_stroke = color_stroke.permute(0, 3, 1, 2)  # [5,128,128,1] -> [5,1,128,128]
        stroke = stroke.view(-1, stork_num, 1, args.width, args.width)  # [5,1,128,128] -> [1,5,1,128,128]
        color_stroke = color_stroke.view(-1, stork_num, 3, args.width, args.width)  # [5b,3,128,128] -> [b,5,3,128,128]
        for i in range(stork_num):
            canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]  # (1 - stroke[:, i]) + color_stroke[:, i]
        return canvas
    '''采用canvas*（1-stroke）+color_stroke的方式，代表1是绘制白色，0是绘制黑色。即能够绘制两种颜色。那么采用的color_class模型就应该在'''
    '''采用canvas*(1-color_stroke)的方式，这时0和1分别表示画笔是否落下，如果输出，代表落下，则绘制出来黑色，如果不落下，则不会绘制'''


    env_batch = 1
    # load_gan(args.resume)
    load_gan("/media/lgf/rl_draw_image/rl_draw_edge_render_change_error/checkpoints/agent_model/Paint_edge-run2")
    import random
    random.seed(100)
    train_loader = getQuickDrawVal(batch_size=env_batch)
    for i, sample_batch in enumerate(train_loader):
        gt = sample_batch.to(torch.uint8).to(device)  # 将gt的数据类型由int64 -> uint8
        gt = gt.expand(env_batch, 3, args.width, args.width)
        show_gt = gt.detach().cpu().numpy().transpose(0, 2, 3, 1)
        # plt.imshow(show_gt[0, :, :, :])
        # plt.show()

        canvas = torch.ones_like(gt)*255
        # show_canvas = canvas.detach().cpu().numpy().transpose(0, 2, 3, 1)
        # plt.imshow(show_canvas[0, :, :, :])
        # plt.show()

        # f = np.random.uniform(0,1,7)
        f = np.array([0.07553265,0.46283505, 0.37771303, 0.47268679, 0.76870261, 0.8379295,
         0.29779711])#长线
        # f = [0.67011608, 0.42159204, 0.27397822 ,0.58364283, 0.68917733, 0.26028718, 0.71636786]#短线
        # print(f)
        # f[-1] = 0.4
        # one_draw = decode(torch.tensor(f,dtype=torch.float).to(device),torch.tensor(canvas/255,dtype=torch.float))

        one_draw = draw_circle_2(f)
        one_draw = torch.tensor(one_draw,dtype=torch.float).to(device)
        one_draw = one_draw.view(1,1,128,128).expand(1,3,128,128)

        gt = gt.float() / 255
        canvas = canvas.float() / 255
        reward_feak = cal_reward(canvas,gt)
        reward_real = cal_reward(gt,gt)
        reward_one_draw = cal_reward(one_draw,gt)
        print('REWARD: real_r_{} |  feak_r-{} | draw_r-{}'.format(reward_real,reward_feak,reward_one_draw))

        plt.subplot(121)
        plt.imshow(show_gt[0, :, :, :])
        plt.subplot(122)
        plt.imshow(one_draw.detach().cpu().numpy().transpose(0, 2, 3, 1)[0,:,:,:])
        plt.show()
        # exit()
        if i > 1:
            break

'''训练gan网络适应线条图'''
# if __name__ == '__main__':
#     env_batch = 64
#     load_gan(args.resume)
#     from Renderer.model import FCN
#     from Renderer.class_model import Class_model
#     Decoder = FCN()  # get [B:1,]
#     Decoder.load_state_dict(torch.load('./checkpoints/renderer_model/render_3/renderer_499999.pth'))
#     color_class = Class_model()
#     color_class.load_state_dict(torch.load('./checkpoints/class_model/class_1_20000.pth'))
#     if torch.cuda.is_available():
#         Decoder = Decoder.cuda()
#         color_class.cuda()
#     train_loader = getQuickDrawTrain(batch_size=env_batch)


