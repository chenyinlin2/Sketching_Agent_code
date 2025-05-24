from matplotlib import pyplot as plt
from utils.var_args import args
from DRL.env import paint_env
from utils.tensorboard import TensorBoard
from DRL.ddpg import DDPG
from DRL.evaluator import Evaluator
import torch
import random
import numpy as np
# train_num = 'test'#记录训练过程中产生结果的类别
# writer = TensorBoard('agent_logs/train_now_log/rl_draw_edge_{}'.format(train_num))
if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    env = paint_env(args.max_step,args.env_batch,None,width=args.width,height=args.height,train_batch=args.train_batch)
    agent = DDPG(args.train_batch, args.env_batch, args.max_step, \
                 args.tau, args.discount, args.rmsize,None,None, args.output,True)
    agent.actor.load_state_dict(torch.load(args.good_actor_check))
    evaluate = Evaluator(args, writer=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        show_all_img = evaluate.test_actor(env, agent.reset, agent.select_action)
        canvas_list = show_all_img['canvas_list']
        target_img = show_all_img['target_img']
        stroke_with_brush_list = show_all_img['stroke_with_brush_list']
        stroke_list = show_all_img['stroke_list']
        plt.imshow(target_img,cmap ='gray')
        plt.show()

        fig1 = plt.figure()
        fig2 = plt.figure()
        fig3 = plt.figure()
        for i in range(len(canvas_list)):
            # stroke_with_brush =
            stroke = stroke_list[i]
            ax1 = fig1.add_subplot(6, 7, i + 1)
            ax1.axis('off')
            ax1.imshow(canvas_list[i],cmap ='gray')
            ax2 = fig2.add_subplot(6, 7, i + 1)
            ax2.axis('off')
            ax2.imshow(stroke_with_brush_list[i],cmap ='gray')

            color_rgb = np.array([[255,64,64]])/255
            # color_stroke = stroke.expand(1,3,args.width,args.width)#[1,3,128,128]
            gt_with_stroke = target_img*(1-stroke)/255 + stroke*color_rgb.reshape(1,1,3)
            ax3 = fig3.add_subplot(6, 7, i + 1)
            ax3.axis('off')
            ax3.imshow(gt_with_stroke,cmap ='gray')
        plt.show()


