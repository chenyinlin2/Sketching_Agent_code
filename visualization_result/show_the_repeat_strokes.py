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

if __name__ == '__main__':
    target_idx = 277
    args.divide = 1
    save_image_output_path_base = '/data/lgf/rl_draw_image/rl_draw_edge_render_start_2_end_new_state/visualization_result/image_output'
    save_image_output_path = os.path.join(save_image_output_path_base,str(target_idx)+'_'+str(args.divide*args.divide))
    save_action_list_path = os.path.join(save_image_output_path,str(target_idx)+'.txt')
    save_image_output_color_stroke_path = os.path.join(save_image_output_path,'color_output')#保存彩色笔划的绘制结果
    os.makedirs(save_image_output_color_stroke_path,exist_ok=True)
    actions_array = np.loadtxt(save_action_list_path)
    print(actions_array.shape)
    # exit()
    # repeat_stroke_idx = 12
    stop_flag_target = 3
    stop_flag = 0  # 提起笔两次，视作停止绘制
    brush_step_num = 0  # 最终绘制使用的步数。
    brush_down_step_num = 0  # 画笔落下绘制的次数
    brush_actions_list = []  # 画笔走过的轨迹
    brush_down_actions_list = []  # 画笔绘制的曲线
    if brush_step_num == 0 and brush_down_step_num == 0:
        for step in range(actions_array.shape[0]):
            brush_actions_list.append(actions_array[step])  # 画笔走过的轨迹
            if actions_array[step, -1] < 0:
                stop_flag += 1
            else:
                brush_down_actions_list.append(actions_array[step])
                stop_flag = 0
            if stop_flag >= stop_flag_target:
                brush_step_num = step - 1
                brush_down_step_num = len(brush_down_actions_list)
                del brush_actions_list[-2:]
                break
            else:
                brush_step_num = step + 1
            if step == actions_array.shape[0] - 1:
                brush_down_step_num = len(brush_down_actions_list)
    elif brush_step_num == 0 or brush_down_step_num == 0:
        for i in range(brush_step_num):
            brush_actions_list.append(actions_array[i])  # 画笔走过的轨迹
            if actions_array[i, -1] > 0:
                brush_down_actions_list.append(actions_array[i])
        brush_down_step_num = len(brush_down_actions_list)
    else:
        for i in range(brush_step_num):
            brush_actions_list.append(actions_array[i])  # 画笔走过的轨迹
            if actions_array[i, -1] > 0:
                brush_down_actions_list.append(actions_array[i])
        if len(brush_down_actions_list) != brush_down_step_num:
            raise ValueError('the brush down num is error')
    print('画笔走过的轨迹的数目', brush_step_num)  # 画笔走过的轨迹的数目
    print('画笔落下的次数', brush_down_step_num)  # 画笔落下的次数


    storkes_color = np.zeros([len(brush_down_actions_list), 3], dtype=float)
    storkes_color[[12,13]] = np.array([1,0,0])# np.tile(np.array([[1,0,0]]),3)

    '''绘制画笔落下的轨迹'''
    brush_down_actions_array = np.array(brush_down_actions_list)
    canvas = np.ones([1, 3, args.width, args.width])
    brush_down_actions_canvas, _ = FreeDecode_one(brush_down_actions_array, canvas, color_expand=storkes_color,
                                                  stork_num=brush_down_step_num)
    brush_down_actions_canvas_show = np.uint8(
        255 * brush_down_actions_canvas.transpose(2, 3, 1, 0)[..., 0])  # B1S0->B0S1
    im = Image.fromarray(brush_down_actions_canvas_show)
    im.save(save_image_output_color_stroke_path + '/{}.jpg'.format('canvas_with_repeat_strokes'))
    fig_bda = plt.figure()
    ax_fig_bda = fig_bda.add_subplot()
    sc2 = ax_fig_bda.imshow(brush_down_actions_canvas_show)
    ax_fig_bda = fig_bda.add_subplot()
    ax_fig_bda.axis('off')
    fig_bda.show()





