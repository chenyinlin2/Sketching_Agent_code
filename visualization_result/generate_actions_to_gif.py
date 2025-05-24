#根据动作数据产生gif图片。
import sys
sys.path.insert(0, r"F:\OneDrive\lgf\研究相关\程序\rl_draw_image\rl_draw_edge_render_start_2_end_new_state")
import copy
import cv2
import numpy as np
import os
import matplotlib as mpl
from utils.var_args import args
from Renderer.stroke_gen import draw_circle_2,draw_circle,draw_circle_2_gif
from matplotlib import pyplot as plt
from PIL import Image
import imageio
# start_point_color = [255, 0, 0]
# def add_color_start_point(stroke,stroke_x):
#     # canvas = np.ones(args.width,args.width)
#     start_point = np.uint8(args.width*stroke_x[0:2])
#     end_point = np.uint8(args.width*stroke_x[4:6])
#     stroke = cv2.circle(stroke, start_point[::-1], 3, start_point_color, -1)
#     stroke = cv2.circle(stroke, end_point[::-1], 3, end_point_color, -1)
#     return stroke
def generate_sketch_gif(x,color_expand=np.ones([1, 3], dtype=float),stork_num = 1,brush_point_color=[255,0,0]):
    x[:,-1] = np.where(x[:,-1] < 0,0,1)
    process_canvas = [] #不带颜色的笔划运动轨迹
    process_with_point_canvas = [] #不带颜色的笔划运动轨迹，带上移动的起点位置
    process_color_canvas = [] #带颜色的笔划运动轨迹
    process_color_with_point_canvas = [] #带颜色的笔划运动轨迹，带上移动的起点位置
    show_down_process_color_with_point_canvas = []#带颜色的笔划运动轨迹，带上移动的起点位置
    show_down_process_with_point_canvas = []#不带颜色的笔划运动轨迹，带上移动的起点位置
    color_canvas = np.ones([args.width, args.width,3])
    canvas = np.ones([args.width, args.width,3])
    show_down_color_canvas = np.ones([args.width, args.width,3])
    show_down_canvas = np.ones([args.width, args.width,3])
    x = x.reshape(-1, args.act_dim + 1)  # [b*stork_num,7]
    small_stroke = []
    if color_expand.shape[0] != x.shape[0]:
        color_expand_ = color_expand.repeat(x.shape[0], axis=0)
    else:
        color_expand_ = color_expand
    for i in range(x.shape[0]):
        _,small_stroke_list = draw_circle_2_gif(x[i,:args.act_dim],args.width,stroke_pixel_length=5)#[B:0,S:1]
        small_stroke.append(small_stroke_list)
    all_small_stroke_canvas = np.empty([0,args.width,args.width,1])#小笔划的矩阵
    all_small_stroke_color_canvas = np.empty([0,args.width,args.width,3])#彩色小笔划的矩阵
    show_down_all_small_stroke_canvas = np.empty([0,args.width,args.width,1])#小笔划的矩阵
    show_down_all_small_stroke_color_canvas = np.empty([0,args.width,args.width,3])#彩色小笔划的矩阵
    all_end_point = np.empty([0,2],dtype=np.uint8)
    for i in range(len(small_stroke)):
        sub_small_stroke_list = small_stroke[i]
        sub_small_stroke_canvas = np.array(sub_small_stroke_list['stroke_canvas'])[...,np.newaxis]#[num,width,width,1] [B:0,S:1]
        end_point = np.array(sub_small_stroke_list['end_point'],dtype=np.uint8)#[num,2]
        sub_color_expand = color_expand_[i].reshape(1,1,1,3)#[1,1,1,3]
        sub_small_stroke_color_canvas = sub_small_stroke_canvas * sub_color_expand#[num,width,width,3]

        all_end_point = np.concatenate((all_end_point,end_point),axis=0)
        all_small_stroke_canvas = np.concatenate((all_small_stroke_canvas,sub_small_stroke_canvas),axis=0)#[nums,128,128,3]
        show_down_all_small_stroke_canvas = np.concatenate((show_down_all_small_stroke_canvas,sub_small_stroke_canvas*x[i,-1]),axis=0)#[nums,128,128,3]
        all_small_stroke_color_canvas = np.concatenate((all_small_stroke_color_canvas,sub_small_stroke_color_canvas),axis=0)#[nums,128,128,3]
        show_down_all_small_stroke_color_canvas = np.concatenate((show_down_all_small_stroke_color_canvas,sub_small_stroke_color_canvas*x[i,-1]),axis=0)#[nums,128,128,3]
    for j in range(all_small_stroke_color_canvas.shape[0]):
        color_canvas = color_canvas * (1-all_small_stroke_canvas[j]) + all_small_stroke_color_canvas[j] #带颜色的笔划
        color_canvas_with_point = copy.copy(color_canvas)*255
        color_canvas_with_point = cv2.circle(color_canvas_with_point, all_end_point[j], 3, brush_point_color, -1)#带颜色笔划，以及起点的笔划
        canvas = canvas * (1-all_small_stroke_canvas[j])#不带颜色的笔划
        canvas_with_point = copy.copy(canvas)*255
        canvas_with_point = cv2.circle(canvas_with_point, all_end_point[j], 3, brush_point_color, -1)#带颜色笔划，以及起点的笔划

        show_down_canvas = show_down_canvas * (1-show_down_all_small_stroke_canvas[j])#不带颜色的笔划
        show_down_canvas_with_point = copy.copy(show_down_canvas)*255
        show_down_canvas_with_point = cv2.circle(show_down_canvas_with_point, all_end_point[j], 3, brush_point_color, -1)#带颜色笔划，以及起点的笔划

        show_down_color_canvas = show_down_color_canvas * (1-show_down_all_small_stroke_canvas[j]) + show_down_all_small_stroke_color_canvas[j] #不带颜色的笔划
        show_down_color_canvas_with_point = copy.copy(show_down_color_canvas)*255
        show_down_color_canvas_with_point = cv2.circle(show_down_color_canvas_with_point, all_end_point[j], 3, brush_point_color, -1)#带颜色笔划，以及起点的笔划

        process_canvas.append(canvas.astype(np.uint8))
        process_color_canvas.append(color_canvas.astype(np.uint8))
        process_with_point_canvas.append(canvas_with_point.astype(np.uint8))
        process_color_with_point_canvas.append(color_canvas_with_point.astype(np.uint8))
        show_down_process_with_point_canvas.append(show_down_canvas_with_point.astype(np.uint8))
        show_down_process_color_with_point_canvas.append(show_down_color_canvas_with_point.astype(np.uint8))
    if x[:,-1].all():
        return process_canvas, process_color_canvas, process_with_point_canvas, process_color_with_point_canvas
    else:
        return process_canvas,process_color_canvas,process_with_point_canvas,process_color_with_point_canvas,show_down_process_with_point_canvas,show_down_process_color_with_point_canvas
if __name__ == '__main__':
    target_idx_all = [2036]
    target_idx_all.sort()
    the_gif_show_time = 4
    for target_idx in target_idx_all:
        args.divide = 1
        dir_name = str(args.divide**2)
        # save_image_output_path_base = '/data/lgf/rl_draw_image/rl_draw_edge_start_2_end_td3/visualization_result/image_output'
        save_image_output_path_base = r'F:\OneDrive\lgf\研究相关\程序\rl_draw_image\rl_draw_edge_render_start_2_end_new_state\visualization_result\image_output'
        save_image_output_path = os.path.join(save_image_output_path_base, dir_name,
                                              str(target_idx) + '_' + str(args.divide * args.divide))
        save_action_list_path = os.path.join(save_image_output_path, str(target_idx) + '.txt')
        save_image_output_color_stroke_path = os.path.join(save_image_output_path, 'color_output')  # 保存彩色笔划的绘制结果
        os.makedirs(save_image_output_color_stroke_path, exist_ok=True)
        actions_array = np.loadtxt(save_action_list_path)
        print(actions_array.shape)

        stop_flag_target = 2
        stop_flag = 0  # 提起笔超过stop_flag_target次，视作停止绘制
        brush_step_num = 28  # 最终绘制使用的步数。
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
            if brush_down_step_num == 0:
                for i in range(brush_step_num):
                    brush_actions_list.append(actions_array[i])  # 画笔走过的轨迹
                    if actions_array[i, -1] > 0:
                        brush_down_actions_list.append(actions_array[i])
                brush_down_step_num = len(brush_down_actions_list)
            elif brush_step_num == 0:
                for i in range(args.max_step):
                    brush_actions_list.append(actions_array[i])  # 画笔走过的轨迹
                    brush_step_num += 1
                    if actions_array[i, -1] > 0:
                        brush_down_actions_list.append(actions_array[i])
                    if len(brush_down_actions_list) == brush_down_step_num:
                        break
        else:
            for i in range(brush_step_num):
                brush_actions_list.append(actions_array[i])  # 画笔走过的轨迹
                if actions_array[i, -1] > 0:
                    brush_down_actions_list.append(actions_array[i])
            if len(brush_down_actions_list) != brush_down_step_num:
                raise ValueError('the brush down num is error')
        print('画笔走过的轨迹的数目', brush_step_num)  # 画笔走过的轨迹的数目
        print('画笔落下的次数', brush_down_step_num)  # 画笔落下的次数

        cmap = mpl.cm.rainbow  # 获取Spectral色条，Spectral_r即为反色
        newcolors = cmap(np.linspace(0, 1, brush_step_num))  # 分片操作
        color_expand = newcolors[:,:3]
        brush_actions_array = np.array(brush_actions_list)
        # 所有笔划 | 带颜色的笔划 | 带有红色起点的笔划 | 带有红色起点和颜色的笔划 | 红色起点的落笔笔划 | 红色起点，带颜色的落笔笔划 |
        process_canvas,process_color_canvas,process_with_point_canvas,process_color_with_point_canvas,show_down_process_with_point_canvas,show_down_process_color_with_point_canvas\
            = generate_sketch_gif(brush_actions_array, color_expand,stork_num=brush_step_num)
        def save_gif(canvas_list,name,show_time):
            save_path = os.path.join(save_image_output_path, str(target_idx) + name)
            gif_fps = np.ceil(len(canvas_list) / show_time)
            gif_fps = 24 if gif_fps < 24 else gif_fps
            add_sub_fps_num = np.int32(gif_fps*show_time - len(canvas_list))
            for _ in range(add_sub_fps_num):
                canvas_list.append(canvas_list[-1])
            imageio.mimsave(save_path, canvas_list, fps=gif_fps)
            print(len(canvas_list),gif_fps)
        save_gif(process_canvas,'_move_trajectory.gif',the_gif_show_time)
        save_gif(process_color_canvas,'_move_color_trajectory.gif',the_gif_show_time)
        save_gif(process_with_point_canvas,'_move_trajectory_with_point.gif',the_gif_show_time)
        save_gif(process_color_with_point_canvas,'_move_color_trajectory_with_point.gif',the_gif_show_time)
        save_gif(show_down_process_with_point_canvas,'_move_trajectory_with_point_only_down.gif',the_gif_show_time)
        save_gif(show_down_process_color_with_point_canvas,'_move_color_trajectory_with_point_only_down.gif',the_gif_show_time)

        cmap = mpl.cm.rainbow  # 获取Spectral色条，Spectral_r即为反色
        newcolors = cmap(np.linspace(0, 1, brush_down_step_num))  # 分片操作
        color_expand = newcolors[:,:3]
        brush_down_actions_array = np.array(brush_down_actions_list)
        canvas = np.ones([1, 3, args.width, args.width])

        process_down_canvas,process_color_down_canvas,process_down_with_point_canvas,process_down_color_with_point_canvas = generate_sketch_gif(brush_down_actions_array, color_expand,stork_num=brush_down_step_num)
        save_gif(process_down_canvas,'_down_move_trajectory.gif',the_gif_show_time)
        save_gif(process_color_down_canvas,'_down_move_trajectory_with_point.gif',the_gif_show_time)
        save_gif(process_down_with_point_canvas,'_down_move_color_trajectory.gif',the_gif_show_time)
        save_gif(process_down_color_with_point_canvas,'_down_move_color_trajectory_with_point.gif',the_gif_show_time)





