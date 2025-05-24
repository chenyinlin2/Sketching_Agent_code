import numpy as np
import os
import matplotlib as mpl
from utils.var_args import args
from Renderer.stroke_gen import draw_circle_2,draw_circle
from matplotlib import pyplot as plt
from PIL import Image

def FreeDecode_one(x,canvas,color_expand=np.ones([1, 3], dtype=float),stork_num = 1): # b * (6 + 1) [1,1,width,width]
    x = x.reshape(-1, args.act_dim + 1)  # [b*stork_num,7]
    # color_x = torch.where(x[:,-1:]>0.,1.,0.)
    # x_copy = x.detach().cpu().numpy()
    stroke = []
    if color_expand.shape[0] != x.shape[0]:
        color_expand_ = color_expand.repeat(x.shape[0], axis=0)
    else:
        print('okk')
        color_expand_ = color_expand
    for i in range(x.shape[0]):
        sub_storke = 1-draw_circle_2(x[i,:args.act_dim],args.width)#[B:1,S:0] -> [B:0,S:1]
        # stroke = np.stack([storke,sub_storke],axis=0) #[b*5,width,width]
        stroke.append(sub_storke)
    stroke = np.array(stroke)
    stroke = stroke.reshape(-1,args.width, args.width, 1)#[b*stork_num,width,width] -> [b*stork_num,width,width,1]
    color_stroke = stroke * color_expand_.reshape(-1, 1, 1, 3) # [b*stork_num,width,width,1]*[b*stork_num,1,1,1] -> [b*stork_num,width,width,1]
    stroke = stroke.transpose(0, 3, 1, 2)#[b*5,width,width,1] -> [b*5,1,width,width]
    color_stroke = color_stroke.transpose(0, 3, 1, 2)#[b*5,width,width,1] -> [b*stork_num,1,width,width]
    stroke = stroke.reshape(-1, stork_num, 1, args.width,args.width)#[b*stork_num,1,width,width] -> [b,stork_num,1,width,width]
    color_stroke = color_stroke.reshape(-1, stork_num, 3, args.width, args.width)#[b*stork_num,1,width,width] -> [b,stork_num,1,width,width]
    for i in range(stork_num):
        canvas = canvas * (1-stroke[:,i]) + color_stroke[:,i] #(1 - stroke[:, i]) + color_stroke[:, i]
    return canvas,(stroke*color_stroke)[0]
def small2large(x):
    # (d * d, width, width) -> (d * width, d * width)
    x = x.reshape(args.divide, args.divide, args.width, args.width, -1)#[16,128,128,3] -> [4,4,128,128,3]
    x = np.transpose(x, (0, 2, 1, 3, 4))#[4,4,128,128,3] -> [4,128,4,128,3]
    x = x.reshape(args.divide * args.width, args.divide * args.width, -1)#[4,128,4,128,3] -> [512,512,3]
    return x
if __name__ == '__main__':
    # target_idx_lists = [51,69,236, 70,74, 152, 175, 174, 298 ,388 ,432 ,476 ,480, 502 ,527 ,539, 666 ,687, 685 ,751]
    target_idx_lists =  [2013,2026,2033,2036,2038,2041,2048,2059,2061,2078,2088,2087,2092,2123,2344,2413]
    target_idx_lists.sort()
    for target_idx in target_idx_lists:
        args.divide = 2
        save_image_output_path_base = '/data/lgf/rl_draw_image/rl_draw_edge_start_2_end_td3/visualization_result/image_output'
        save_image_output_path = os.path.join(save_image_output_path_base,str(args.divide*args.divide),str(target_idx)+'_'+str(args.divide*args.divide))
        save_action_list_path = os.path.join(save_image_output_path,str(target_idx)+'.txt')
        save_image_output_color_stroke_path = os.path.join(save_image_output_path,'color_output')#保存彩色笔划的绘制结果
        os.makedirs(save_image_output_color_stroke_path,exist_ok=True)
        actions_array = np.loadtxt(save_action_list_path)
        print(actions_array.shape)
        # exit()

        brush_step_num = 0  # 最终绘制使用的步数。
        brush_down_step_num = 0  # 画笔落下绘制的次数
        brush_actions_list = []  # 画笔走过的轨迹
        brush_down_actions_list = []  # 画笔绘制的曲线
        for sub_image_idx in range(args.divide*args.divide):
            sub_actions_array = actions_array[40*sub_image_idx:40*sub_image_idx+40]
            stop_flag_target = 3
            stop_flag = 0  # 提起笔超过stop_flag_target次，视作停止绘制
            sub_brush_step_num = 0  # 最终绘制使用的步数。
            sub_brush_down_step_num = 0  # 画笔落下绘制的次数
            sub_brush_actions_list = []  # 画笔走过的轨迹
            sub_brush_down_actions_list = []  # 画笔绘制的曲线
            if sub_brush_step_num == 0 and sub_brush_down_step_num == 0:
                for step in range(sub_actions_array.shape[0]):
                    sub_brush_actions_list.append(sub_actions_array[step])  # 画笔走过的轨迹
                    if sub_actions_array[step, -1] < 0:
                        stop_flag += 1
                    else:
                        sub_brush_down_actions_list.append(sub_actions_array[step])
                        stop_flag = 0
                    if stop_flag >= stop_flag_target:
                        sub_brush_step_num = step - 1
                        sub_brush_down_step_num = len(sub_brush_down_actions_list)
                        del sub_brush_actions_list[-2:]
                        break
                    else:
                        sub_brush_step_num = step + 1
                    if step == sub_actions_array.shape[0] - 1:
                        sub_brush_down_step_num = len(sub_brush_down_actions_list)
            elif sub_brush_step_num == 0 or sub_brush_down_step_num == 0:
                if sub_brush_down_step_num == 0:
                    for i in range(sub_brush_step_num):
                        sub_brush_actions_list.append(sub_actions_array[i])  # 画笔走过的轨迹
                        if sub_actions_array[i, -1] > 0:
                            sub_brush_down_actions_list.append(sub_actions_array[i])
                    sub_brush_down_step_num = len(sub_brush_down_actions_list)
                elif sub_brush_step_num == 0:
                    for i in range(args.max_step):
                        sub_brush_actions_list.append(sub_actions_array[i])  # 画笔走过的轨迹
                        sub_brush_step_num += 1
                        if sub_actions_array[i, -1] > 0:
                            sub_brush_down_actions_list.append(sub_actions_array[i])
                        if len(sub_brush_down_actions_list) == sub_brush_down_step_num:
                            break
            else:
                for i in range(sub_brush_step_num):
                    sub_brush_actions_list.append(sub_actions_array[i])  # 画笔走过的轨迹
                    if sub_actions_array[i, -1] > 0:
                        sub_brush_down_actions_list.append(sub_actions_array[i])
                if len(sub_brush_down_actions_list) != sub_brush_down_step_num:
                    raise ValueError('the brush down num is error')
            brush_step_num += sub_brush_step_num
            brush_down_step_num += sub_brush_down_step_num
            brush_actions_list.append(sub_brush_actions_list)
            brush_down_actions_list.append(sub_brush_down_actions_list)
        print('画笔走过的轨迹的数目', brush_step_num)  # 画笔走过的轨迹的数目
        print('画笔落下的次数', brush_down_step_num)  # 画笔落下的次数
        # exit()

        # print(newcolors)
        '''绘制画笔走过的轨迹'''
        cmap = mpl.cm.rainbow  # 获取Spectral色条，Spectral_r即为反色
        newcolors = cmap(np.linspace(0, 1, brush_step_num))  # 分片操作
        color_expand = newcolors[:,:3]#[41,3]
        left_color = color_expand
        brush_actions_canvas_list = []
        for sub_idx in range(len(brush_actions_list)):
            sub_brush_actions_array = np.array(brush_actions_list[sub_idx])
            sub_brush_actions_num = sub_brush_actions_array.shape[0]
            sub_color_expand = left_color[:sub_brush_actions_num,:]
            left_color = left_color[sub_brush_actions_num:,:]
            canvas = np.ones([1, 3, args.width, args.width])
            if sub_brush_actions_num != 0:
                sub_brush_actions_canvas, _ = FreeDecode_one(sub_brush_actions_array, canvas, color_expand=sub_color_expand,
                                                     stork_num=sub_brush_actions_num)
            else:
                sub_brush_actions_canvas = canvas
            # print(sub_brush_actions_canvas.shape)
            brush_actions_canvas_list.append(sub_brush_actions_canvas)

            # brush_actions_canvas_show = np.uint8(255 * sub_brush_actions_canvas.transpose(2, 3, 1, 0)[..., 0])  # B1S0->B0S1
            # fig_ba = plt.figure()
            # ax_fig_ba = fig_ba.add_subplot()
            # sc1 = ax_fig_ba.imshow(brush_actions_canvas_show, cmap=cmap)
            # fig_ba.show()
        brush_actions_canvas_array = np.array(brush_actions_canvas_list).squeeze(1).transpose(0,2,3,1)
        big_brush_actions_canvas = small2large(brush_actions_canvas_array)
        print(big_brush_actions_canvas.shape)

        brush_actions_canvas_show = np.uint8(255*big_brush_actions_canvas)#B1S0->B0S1
        im = Image.fromarray(brush_actions_canvas_show)
        im.save(save_image_output_color_stroke_path+'/{}.jpg'.format('brush_trajectory'))
        '''显示图片'''
        fig_ba = plt.figure()
        ax_fig_ba = fig_ba.add_subplot()
        sc1 = ax_fig_ba.imshow(brush_actions_canvas_show, cmap=cmap)
        fig_ba.savefig(save_image_output_color_stroke_path+'/{}.svg'.format('brush_trajectory'), format='svg', dpi=200)  # 输出

        ax_fig_ba.axis('off')
        cb1 = plt.colorbar(sc1, aspect=50)
        cb1.set_ticks([])  # 颜色条上的数字不显示
        fig_ba.show()
        fig_ba.savefig(save_image_output_color_stroke_path+'/{}_with_bar.svg'.format('brush_trajectory'), format='svg', dpi=200)  # 输出
        '''设计绘制颜色'''
        newcolors = cmap(np.linspace(0, 1, brush_down_step_num))  # 分片操作
        # newcolors =cm.get_cmap(cmap)(np.linspace(0, 1, brush_down_step_num))
        color_expand = newcolors[:, :3]
        left_color = color_expand
        brush_down_actions_canvas_list = []
        '''绘制画笔落下的轨迹'''
        for sub_idx in range(len(brush_down_actions_list)):
            sub_brush_down_actions_array = np.array(brush_down_actions_list[sub_idx])
            canvas = np.ones([1, 3, args.width, args.width])
            sub_brush_down_actions_num = sub_brush_down_actions_array.shape[0]
            sub_color_expand = left_color[:sub_brush_down_actions_num,:]
            left_color = left_color[sub_brush_down_actions_num:,:]
            if sub_brush_down_actions_num != 0:
                sub_brush_down_actions_canvas, _ = FreeDecode_one(sub_brush_down_actions_array, canvas, color_expand=sub_color_expand,
                                                      stork_num=sub_brush_down_actions_num)
            else:
                sub_brush_down_actions_canvas = canvas
            brush_down_actions_canvas_list.append(sub_brush_down_actions_canvas)
        brush_down_actions_canvas_array = np.array(brush_down_actions_canvas_list).squeeze(1).transpose(0,2,3,1)
        big_brush_down_actions_canvas = small2large(brush_down_actions_canvas_array)
        print(big_brush_down_actions_canvas.shape)
        # print(brush_down_actions_canvas.shape)
        brush_down_actions_canvas_show = np.uint8(255*big_brush_down_actions_canvas)#B1S0->B0S1
        im = Image.fromarray(brush_down_actions_canvas_show)
        im.save(save_image_output_color_stroke_path+'/{}.jpg'.format('brush_down_trajectory'))
        fig_bda = plt.figure()
        ax_fig_bda = fig_bda.add_subplot()
        sc2 = ax_fig_bda.imshow(brush_down_actions_canvas_show, cmap=cmap)
        ax_fig_bda.axis('off')
        fig_bda.savefig(save_image_output_color_stroke_path+'/{}.svg'.format('brush_down_trajectory'), format='svg', dpi=200)  # 输出
        cb2 = plt.colorbar(sc2, aspect=50)  # aspect控制bar的宽度
        cb2.set_ticks([])  # 颜色条上的数字不显示
        fig_bda.show()
        fig_bda.savefig(save_image_output_color_stroke_path+'/{}_with_bar.svg'.format('brush_down_trajectory'), format='svg', dpi=200)  # 输出


