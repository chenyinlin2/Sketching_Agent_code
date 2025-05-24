import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 确定使用哪一块gpu
from matplotlib import pyplot as plt
from utils.var_args import args
import torch
import random
import numpy as np
from utils.chamfer_distance import chamfer_distance_numpy
from utils.reconstruct_complexity import get_bezier_length
writer = None
def save_mean_para(para_lists,path,name):
    mean_para = np.mean(para_lists)
    para_lists.append(mean_para)
    para_array = np.array(para_lists)
    save_para_path = os.path.join(path, name)
    np.savetxt(save_para_path, para_array)
def read_txt(path):
    try:
        # 打开文件以供读取
        with open(path, 'r') as file:
            # 读取文件的每一行并将其拆分成元素，然后存储在二维列表中
            lines = [line.strip().split() for line in file]
    except FileNotFoundError:
        print("文件未找到")
    except Exception as e:
        print(f"发生错误: {e}")
    return lines
if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.env_batch = 1
    args.var_thick_flag = False
    args.train_fune_flag = False
    target_img_size = 128
    save_output_path_base = '/data/lgf/rl_draw_image/rl_draw_edge_render_start_2_end_new_state/quantitative_indicators_output/'+str(target_img_size)+'/run_5'
    os.makedirs(save_output_path_base,exist_ok=True)#保存时间，dis，笔划等信息。
    save_actions_output_path_base = os.path.join(save_output_path_base,'actions_para_list')
    os.makedirs(save_actions_output_path_base,exist_ok=True)#将动作信息保存为txt文件
    # save_image_output_path_base = '/data/lgf/rl_draw_image/rl_draw_edge_start_2_end_td3/visualization_result/image_output_fune'
    reconstruct_complexity_lists = []  # 重构复杂度
    brush_down_length_lists = []  # 画笔落下过程中智能体完成的长度
    brush_length_lists = []  # 智能体一共完成的长度
    brush_step_num_lists = []  # 笔划数目列表
    brush_down_step_num_lists = []  # 笔划落下数目列表
    charmfer_dis_lists = []  # 倒角距离列表
    with torch.no_grad():
        for idx in range(5000):
            # gt = sample_batch.to(torch.uint8)
            # gt = gt.expand(gt.shape[0], 3, args.width, args.width)
            read_action_list_path = os.path.join(save_actions_output_path_base, str(idx) + '.txt')
            action_list = read_txt(read_action_list_path)
            '''开始计算'''
            stop_flag_target = 3  # 超过三次没有绘制笔划，认为停止运算了。
            stop_flag = 0  # 提起笔两次，视作停止绘制
            brush_step_num = 0  # 最终绘制使用的步数。
            brush_down_step_num = 0  # 画笔落下绘制的次数
            brush_length = 0  # 智能体总共走的长度
            brush_down_length = 0  # 画笔落下状态下智能体走的长度
            for action_idx in range(args.max_step):
                action = np.array([float(s) for s in action_list[action_idx]])
                if stop_flag < stop_flag_target:  # 没有停止绘制
                    '''计算相关的笔划数目'''
                    brush_step_num += 1
                    if action[-1] > 0:
                        brush_down_step_num += 1
                        stop_flag = 0
                    else:
                        stop_flag += 1
                    if stop_flag < stop_flag_target:
                        curve_length = get_bezier_length(action[0:2], action[2:4], action[4:6])
                        brush_length += curve_length
                        brush_down_length += curve_length * (0. if action[-1] < 0 else 1.)
            brush_length_lists.append(brush_length)
            brush_down_length_lists.append(brush_down_length)
            reconstruct_complexity_lists.append(brush_down_length / brush_length)
            print(str(idx) + 'is over')
        '''保存智能体落下时绘制长度，智能体总共的长度，重构复杂度。'''
        # brush_length_lists.append(sum(brush_length_lists) / len(brush_length_lists))
        # brush_down_length_lists.append(sum(brush_down_length_lists) / len(brush_down_length_lists))
        # reconstruct_complexity_lists.append(sum(reconstruct_complexity_lists) / len(reconstruct_complexity_lists))
        save_mean_para(brush_length_lists, save_output_path_base, 'brush_length_lists.txt')
        save_mean_para(brush_down_length_lists, save_output_path_base, 'brush_down_length_lists.txt')
        save_mean_para(reconstruct_complexity_lists, save_output_path_base, 'reconstruct_complexity_lists.txt')