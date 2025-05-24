import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 确定使用哪一块gpu
from matplotlib import pyplot as plt
from utils.var_args import args
from DRL.env import paint_env
# from utils.tensorboard import TensorBoard
from tensorboardX import SummaryWriter
from DRL.ddpg import DDPG
from DRL.evaluator import Evaluator
import torch
import random
import numpy as np
from PIL import Image
import time
from utils.chamfer_distance import chamfer_distance_numpy
writer = None
def save_mean_para(para_lists,path,name):
    mean_para = np.mean(para_lists)
    para_lists.append(mean_para)
    para_array = np.array(para_lists)
    save_para_path = os.path.join(path, name)
    np.savetxt(save_para_path, para_array)
if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.env_batch = 1
    args.var_thick_flag = False
    args.train_fune_flag = False
    target_img_size = 128
    save_output_path_base = '/data/lgf/rl_draw_image/rl_draw_edge_render_start_2_end_new_state/quantitative_indicators_output/'+str(target_img_size)+'/run_1'
    os.makedirs(save_output_path_base,exist_ok=True)#保存时间，dis，笔划等信息。
    save_actions_output_path_base = os.path.join(save_output_path_base,'actions_para_list')
    os.makedirs(save_actions_output_path_base,exist_ok=True)#将动作信息保存为txt文件
    # save_image_output_path_base = '/data/lgf/rl_draw_image/rl_draw_edge_start_2_end_td3/visualization_result/image_output_fune'

    env = paint_env(args.max_step,args.env_batch,writer,width=args.width,height=args.height,train_batch=args.train_batch)
    agent = DDPG(args.train_batch, args.env_batch, args.max_step, \
                 args.tau, args.discount, args.rmsize,None,None, args.output,True)
    agent.actor.load_state_dict(torch.load(args.good_actor_check))
    evaluate = Evaluator(args, writer=writer,shuffle=False)
    use_time_lists = []#用时
    brush_step_num_lists = []#笔划数目列表
    brush_down_step_num_lists = []#笔划落下数目列表
    charmfer_dis_lists = []#倒角距离列表
    with torch.no_grad():
        for idx, sample_batch in enumerate(evaluate.val_loader):
            gt = sample_batch.to(torch.uint8)
            gt = gt.expand(gt.shape[0], 3, args.width, args.width)
            charmfer_input_A = gt.detach().numpy()[0]/255
            '''开始计算'''
            episode_steps = 0
            stop_flag_target = 3  # 超过三次没有绘制笔划，认为停止运算了。
            stop_flag = 0  # 提起笔两次，视作停止绘制
            brush_step_num = 0  # 最终绘制使用的步数。
            brush_down_step_num = 0  # 画笔落下绘制的次数
            actions_list = [[] for _ in range(gt.shape[0])]  # 绘制过程中画笔落下的笔划
            observation = env.reset(gt)  # 范围是[0-255
            assert observation is not None
            start_time = time.time()#开始计算时间
            while (episode_steps < args.max_step or not args.max_step):
                action = agent.select_action(observation)  # [batch,7]
                observation, reward, done, next_canvas, stroke_with_brush, stroke = env.step_test(
                    action)  # stroke_with_brush相当于画笔移动的轨迹，stroke只是画笔绘制的笔划。
                episode_steps += 1
                for sample_idx in range(gt.shape[0]):
                    actions_list[sample_idx].append(action[sample_idx])  # 保存所有笔划
                if stop_flag < stop_flag_target:#没有停止绘制
                    '''计算相关的笔划数目'''
                    brush_step_num += 1
                    if action[0,-1] > 0:
                        brush_down_step_num += 1
                        stop_flag = 0
                    else:
                        stop_flag += 1
                    if stop_flag >= stop_flag_target:
                        end_time = time.time()#计算结束时间
            charmfer_input_B = next_canvas.detach().cpu().numpy()[0]/255
            if stop_flag < stop_flag_target:
                end_time = time.time()
            use_time = end_time-start_time
            use_time_lists.append(use_time)
            brush_step_num_lists.append(brush_step_num)
            brush_down_step_num_lists.append(brush_down_step_num)
            charmfer_dis = chamfer_distance_numpy(charmfer_input_A,charmfer_input_B)
            charmfer_dis_lists.append(charmfer_dis)
            '''保存actions_list'''
            actions_list_save = []
            args.divide = int(np.sqrt(len(actions_list)))
            for i in range(len(actions_list)):
                actions_list_save.append(np.array(actions_list[i]))
            actions_list_save_array = np.array(actions_list_save).reshape(-1, 7)
            save_action_list_path = os.path.join(save_actions_output_path_base, str(idx) + '.txt')
            np.savetxt(save_action_list_path, actions_list_save_array)
            print(str(idx)+'is over')
        '''保存时间，dis，笔划数目信息。'''
        save_mean_para(use_time_lists,save_output_path_base,'use_time.txt')
        save_mean_para(brush_step_num_lists,save_output_path_base,'brush_step_nums.txt')
        save_mean_para(brush_down_step_num_lists,save_output_path_base,'brush_down_step_nums.txt')
        save_mean_para(charmfer_dis_lists,save_output_path_base,'charmfer_dis.txt')







