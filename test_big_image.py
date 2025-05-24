import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 确定使用哪一块gpu
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
def small2large(x):
    # (d * d, width, width) -> (d * width, d * width)
    x = x.reshape(args.divide, args.divide, args.width, args.width, -1)#[16,128,128,3] -> [4,4,128,128,3]
    x = np.transpose(x, (0, 2, 1, 3, 4))#[4,4,128,128,3] -> [4,128,4,128,3]
    x = x.reshape(args.divide * args.width, args.divide * args.width, -1)#[4,128,4,128,3] -> [512,512,3]
    return x
writer = None
if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.env_batch = 1
    args.var_thick_flag = False
    args.var_patch_image = True
    args.train_fune_flag = False
    sub_image_width = 100
    save_image_output_path_base = '/data/lgf/rl_draw_image/rl_draw_edge_render_start_2_end_new_state/visualization_result/image_output'
    os.makedirs(save_image_output_path_base,exist_ok=True)
    # save_image_output_path_base = '/data/lgf/rl_draw_image/rl_draw_edge_start_2_end_td3/visualization_result/image_output_fune'

    env = paint_env(args.max_step,args.env_batch,writer,width=args.width,height=args.height,train_batch=args.train_batch)
    agent = DDPG(args.train_batch, args.env_batch, args.max_step, \
                 args.tau, args.discount, args.rmsize,None,None, args.output,True)
    agent.actor.load_state_dict(torch.load(args.good_actor_check))
    evaluate = Evaluator(args, writer=writer,shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # target_idx_all = [51,69,236, 70,74, 152, 175, 174, 298 ,388 ,432 ,476 ,480, 502 ,527 ,539, 666 ,687, 685 ,751]
    # target_idx_all = [1014 ,1046 ,1061, 1344, 1529]
    # target_idx_all = [86,82 ,126, 151 ,152 ,174 ,211 ,242, 254,  266, 277, 278]
    # target_idx_all = [2013,2026,2033,2036,2038,2041,2048,2059,2061,2078,2088,2087,2092,2123,2344,2413]
    target_idx_all = [2092]
    target_idx_all.sort()
    for target_idx in target_idx_all:
        with torch.no_grad():
            show_all_img,actions_list = evaluate.test_actor_with_var_thick(env, agent.reset, agent.select_action,target_idx=target_idx)
        if writer != None:
            writer.close()
        # print(actions_list)
        actions_list_save = []
        args.divide = int(np.sqrt(len(actions_list)))
        save_image_output_path = os.path.join(save_image_output_path_base,str(len(actions_list)),str(target_idx)+'_'+str(len(actions_list)))
        os.makedirs(save_image_output_path,exist_ok=True)
        print(args.divide)
        '''将actions保存到txt文件中'''
        for i in range(len(actions_list)):
            actions_list_save.append(np.array(actions_list[i]))
        actions_list_save_array = np.array(actions_list_save).reshape(-1,7)
        save_var = '\n'
        save_action_list_path = os.path.join(save_image_output_path,str(target_idx)+'.txt')
        np.savetxt(save_action_list_path,actions_list_save_array)
        print(actions_list_save_array.shape)
        # exit()
        '''将产生的图片结果保存'''
        canvas_list = show_all_img['canvas_list']#[args.d*args.d,40,128,128,3]
        target_img = show_all_img['target_img']#[args.d*args.d,128,128,3]
        stroke_with_brush_list = show_all_img['stroke_with_brush_list']#[args.d*args.d,40,128,128,3]
        stroke_list = show_all_img['stroke_list']#[args.d*args.d,40,128,128,1]
        save_image_output_gt_path = os.path.join(save_image_output_path,'gt.jpg')#保存真值图
        target_img_save = small2large(np.array(target_img))
        im = Image.fromarray(target_img_save)
        im.save(save_image_output_gt_path)
        save_image_output_process_path_base = os.path.join(save_image_output_path,'process_'+str(target_idx)+'_'+str(args.divide*args.divide))
        os.makedirs(save_image_output_process_path_base,exist_ok=True)
        canvas_list_array = np.array(canvas_list)#list -> [args.d*args.d,40,128,128,3]
        for i in range(args.max_step):
            sub_canvas = canvas_list_array[:,i,...]#B1S0->B0S1
            save_image_output_process_path = os.path.join(save_image_output_process_path_base,str(i)+'.jpg')
            sub_canvas_big = small2large(sub_canvas)
            im = Image.fromarray(sub_canvas_big)
            im.save(save_image_output_process_path)
        save_image_output_stroke_process_path_base = os.path.join(save_image_output_path,'stroke_process_'+str(target_idx)+'_'+str(args.divide*args.divide))
        os.makedirs(save_image_output_stroke_process_path_base,exist_ok=True)
        stroke_list_array = np.array(stroke_list)#list -> [args.d*args.d,40,128,128,3]
        for i in range(args.max_step):
            sub_stroke = stroke_list_array[:,i,...]
            save_image_output_process_path = os.path.join(save_image_output_stroke_process_path_base,str(i)+'.png')
            sub_stroke_big = np.uint8(255-np.tile(small2large(sub_stroke),(1,3))*255)
            im = Image.fromarray(sub_stroke_big)
            im.save(save_image_output_process_path)



