#在论文中，我们需要每个笔划的初始点和终止点，使用红色和黄色点进行标记。
import numpy as np
import os
import matplotlib as mpl
from utils.var_args import args
from Renderer.stroke_gen import draw_circle_2,draw_circle
from matplotlib import pyplot as plt
from PIL import Image
import cv2
end_point_color = [253, 191, 45]
start_point_color = [255, 0, 0]
def add_color_start_point(stroke,stroke_x):
    # canvas = np.ones(args.width,args.width)
    start_point = np.uint8(args.width*stroke_x[0:2])
    end_point = np.uint8(args.width*stroke_x[4:6])
    stroke = cv2.circle(stroke, start_point[::-1], 3, start_point_color, -1)
    stroke = cv2.circle(stroke, end_point[::-1], 3, end_point_color, -1)
    return stroke

def FreeDecode_one(x,canvas,color_expand=np.zeros([1, 3], dtype=float),stork_num = 1): # b * (6 + 1) [1,1,width,width]
    x = x.reshape(-1, args.act_dim + 1)  # [b*stork_num,7]
    stroke_with_start_end_point = []
    stroke = []
    sub_canvas_list = []
    if color_expand.shape[0] != x.shape[0]:
        color_expand_ = color_expand.repeat(x.shape[0], axis=0)
    else:
        print('okk')
        color_expand_ = color_expand
    for i in range(x.shape[0]):
        sub_storke = 1-draw_circle_2(x[i,:args.act_dim],args.width)#[B:1,S:0] -> [B:0,S:1]
        sub_storke_input = np.tile(1-sub_storke[...,None],(1,1,3))*255#[B:0,S:1] -> [B:1,S:0]
        stroke_with_start_end_point.append(add_color_start_point(sub_storke_input,x[i]))
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
        sub_canvas_list.append(canvas[0])
    return canvas,np.array(stroke_with_start_end_point,dtype=np.uint8),np.array(sub_canvas_list,dtype=np.float32)

if __name__ == '__main__':
    target_idx = 277
    args.divide = 1
    save_image_output_path_base = '/data/lgf/rl_draw_image/rl_draw_edge_render_start_2_end_new_state/visualization_result/image_output'
    save_image_output_path = os.path.join(save_image_output_path_base,
                                          str(target_idx) + '_' + str(args.divide * args.divide))
    save_action_list_path = os.path.join(save_image_output_path, str(target_idx) + '.txt')
    save_image_stroke_color_sta_end_path = os.path.join(save_image_output_path, 'stroke_with_start_end_output')  # 保存彩色笔划的绘制结果
    os.makedirs(save_image_stroke_color_sta_end_path, exist_ok=True)
    actions_array = np.loadtxt(save_action_list_path)
    print(actions_array.shape)
    # exit()
    stop_flag_target = 3
    stop_flag = 0  # 提起笔stop_flag_target次，视作停止绘制
    brush_step_num = 0  # 最终绘制使用的步数。
    brush_down_step_num = 0  # 画笔落下绘制的次数
    brush_actions_list = []  # 画笔走过的轨迹
    brush_down_actions_list = []  # 画笔绘制的曲线
    '''计算这个过程中实际落下的笔划数目'''
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
        raise ValueError('one various is zero,which is not allowed')
    else:
        for i in range(brush_step_num):
            brush_actions_list.append(actions_array[i])  # 画笔走过的轨迹
            if actions_array[i, -1] > 0:
                brush_down_actions_list.append(actions_array[i])
        if len(brush_down_actions_list) != brush_down_step_num:
            raise ValueError('the brush down num is error')
    print('画笔走过的轨迹的数目', brush_step_num)  # 画笔走过的轨迹的数目
    print('画笔落下的次数', brush_down_step_num)  # 画笔落下的次数

    brush_actions_array = np.array(brush_actions_list)
    canvas = np.ones([1, 3, args.width, args.width])
    brush_actions_canvas, stroke_with_start_end_point,sub_canvas_array = FreeDecode_one(brush_actions_array, canvas,
                                             stork_num=brush_step_num)
    '''保存笔划，起点和重点标注颜色'''
    print(stroke_with_start_end_point.shape)
    for sub_idx in range(brush_step_num):
        sub_save_image_stroke_color_sta_end_path = os.path.join(save_image_stroke_color_sta_end_path,str(sub_idx)+'.jpg')
        sub_stroke_with_start_end_img = stroke_with_start_end_point[sub_idx]
        im = Image.fromarray(sub_stroke_with_start_end_img)
        im.save(sub_save_image_stroke_color_sta_end_path)
    '''保存绘制过程中的图片，起点和重点标注颜色'''
    save_sub_canvas_stroke_sta_end_path = os.path.join(save_image_output_path, 'sub_canvas_with_start_end_output')  # 保存彩色笔划的绘制结果
    os.makedirs(save_sub_canvas_stroke_sta_end_path, exist_ok=True)
    # print(stroke_with_start_end_point.shape)
    for sub_canvas_idx in range(brush_step_num):
        sub_canvas = sub_canvas_array[sub_canvas_idx].transpose(1,2,0)*255
        sub_save_image_stroke_color_sta_end_path = os.path.join(save_sub_canvas_stroke_sta_end_path,str(sub_canvas_idx)+'.jpg')
        sub_canvas_save = add_color_start_point(sub_canvas.copy(),brush_actions_array[sub_canvas_idx])
        im = Image.fromarray(np.uint8(sub_canvas_save))
        im.save(sub_save_image_stroke_color_sta_end_path)

    '''绘制画笔下落中的相关图片'''
    brush_down_actions_array = np.array(brush_down_actions_list)
    canvas = np.ones([1, 3, args.width, args.width])
    brush_down_actions_canvas,down_stroke_with_start_end_point ,sub_canvas_down_array = FreeDecode_one(brush_down_actions_array, canvas,
                                                  stork_num=brush_down_step_num)
    '''保存笔划，起点和重点标注颜色'''
    print(brush_down_actions_canvas.shape)
    save_image_stroke_color_sta_end_path = os.path.join(save_image_output_path, 'down_stroke_with_start_end_output')  # 保存彩色笔划的绘制结果
    os.makedirs(save_image_stroke_color_sta_end_path, exist_ok=True)
    for sub_idx in range(brush_down_step_num):
        sub_save_image_stroke_color_sta_end_path = os.path.join(save_image_stroke_color_sta_end_path,
                                                                str(sub_idx) + '.jpg')
        sub_stroke_with_start_end_img = stroke_with_start_end_point[sub_idx]
        im = Image.fromarray(sub_stroke_with_start_end_img)
        im.save(sub_save_image_stroke_color_sta_end_path)
    '''保存绘制过程中的图片，起点和重点标注颜色'''
    save_sub_canvas_stroke_sta_end_path = os.path.join(save_image_output_path,
                                                       'sub_canvas_with_down_start_end_output')  # 保存彩色笔划的绘制结果
    os.makedirs(save_sub_canvas_stroke_sta_end_path, exist_ok=True)
    # print(stroke_with_start_end_point.shape)
    for sub_canvas_idx in range(brush_down_step_num):
        sub_canvas = sub_canvas_down_array[sub_canvas_idx].transpose(1, 2, 0) * 255
        sub_save_image_stroke_color_sta_end_path = os.path.join(save_sub_canvas_stroke_sta_end_path,
                                                                str(sub_canvas_idx) + '.jpg')
        sub_canvas_save = add_color_start_point(sub_canvas.copy(), brush_down_actions_array[sub_canvas_idx])
        im = Image.fromarray(np.uint8(sub_canvas_save))
        im.save(sub_save_image_stroke_color_sta_end_path)