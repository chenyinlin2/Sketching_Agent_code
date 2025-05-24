'''每次重新运行，需要检查 | writer | args.output | '''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 确定使用哪一块gpu
import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from utils.tensorboard import TensorBoard
import time
from utils.util import *
from DRL.env import paint_env
from dataloader.mnist_loader.mnist_128_loader import getMnistTrain,getMnistTest
from dataloader.QuickDraw_clean.var_shape_quickdraw import getQuickDrawTrain
from DRL.ddpg import DDPG
from DRL.evaluator import Evaluator
train_num = 8#记录训练过程中产生结果的类别
writer = TensorBoard('agent_logs/train_now_log/rl_draw_edge_{}'.format(train_num))

def train(env:paint_env,agent:DDPG,evaluate:Evaluator):
    train_times = args.train_times
    env_batch = args.env_batch
    train_batch = args.train_batch
    max_step = args.max_step
    debug = args.debug
    episode_train_times = args.episode_train_times#10
    validate_interval = args.validate_interval
    output_path = args.output#输出路径
    time_stamp = time.time()
    episode = 0
    episode_steps = 0
    step = 0
    tot_reward = 0.
    observation = None
    noise_factor = args.noise_factor#0 动作上面是否增加噪声 扩大搜索范围。
    var_thick_flag = args.var_thick_flag#是否使用不同宽度的画笔绘制不同大小的图片的标志位
    train_loader = getQuickDrawTrain(batch_size=env_batch,var_thick_flag=var_thick_flag)
    validate_reward_max = -100
    validate_dist_max = -100
    while step <= train_times:
        for i,sample_batch in enumerate(train_loader):
            if var_thick_flag:
                del_white_num = 0
                for sub_idx in range(len(sample_batch)):#删掉其中的空白图像。即全是白色的图像
                    sub_stroke_batch = sample_batch[sub_idx - del_white_num]
                    all_white_flag = np.all(sub_stroke_batch == 255)# 判断是否全部元素是255
                    if all_white_flag:
                        del sample_batch[sub_idx - del_white_num]
                        del_white_num += 1
                if del_white_num > 0:
                    sample_batch.append(np.ones_like(sample_batch[0])*255)
                sample_batch = torch.tensor(np.array(sample_batch), dtype=torch.float)

            gt = sample_batch.to(torch.uint8)#将gt的数据类型由int64 -> uint8
            cur_env_batch = gt.shape[0]#env_batch被切割成了cur_env_batch
            gt = gt.expand(cur_env_batch,3,args.width,args.width)
            done = np.array([False] * cur_env_batch)
            if episode > 2000:#当超过一定episode之后，改变train_batch
                agent.train_batch = 96
            while not done.all():
                step += 1
                episode_steps += 1#当前轨迹已走的步数，最大步数是max_step
                if observation is None:
                    observation = env.reset(gt)#范围是[0-255] canvas, img, T #[batch,1,width,width] | [batch,1,width,width] | [batch,1,width,width](tensor)
                    agent.reset(observation, noise_factor,step)
                action = agent.select_action(observation, noise_factor=noise_factor)# [batch,args.act_dim+1] (numpy)
                observation, reward, done, _ = env.step(action)#(tensor:cuda)(numpy:cpu)(numpy:cpu)
                agent.observe(reward, observation, done, step)
                # reward, dist = evaluate(env, agent.select_action, debug=debug)
                if (episode_steps >= max_step and max_step):
                    if step > args.warmup:
                #         # [optional] evaluate
                      if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                            reward, dist,charmfer_dist = evaluate(env, agent.reset,agent.select_action, debug=debug)
                            if debug: prRed(
                                'Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'.format(step - 1,
                                                                                                          np.mean(reward),
                                                                                                          np.mean(dist),
                                                                                                          np.var(dist)))
                            writer.add_scalar('validate/mean_reward', np.mean(reward), step)
                            writer.add_scalar('validate/mean_dist', np.mean(dist), step)
                            writer.add_scalar('validate/var_dist', np.var(dist), step)
                            writer.add_scalar('validate/charmfer_dist', charmfer_dist, step)

                            if np.mean(reward) > validate_reward_max:
                                validate_reward_max = np.mean(reward)
                                agent.save_model(output_path,'best')
                                print('saved the max validate reward model')
                            elif np.mean(dist) > validate_dist_max:
                                validate_dist_max = np.mean(dist)
                                agent.save_model(output_path,'best_dist_')
                                print('saved the max validate dist model')
                            else:
                                agent.save_model(output_path)
                    train_time_interval = time.time() - time_stamp#两次训练之间的间隔
                    time_stamp = time.time()
                    tot_Q = 0.
                    tot_value_loss = 0.
                    if step > args.warmup:
                        if step < 10000 * max_step:
                            lr = (3e-4, 1e-3)
                        elif step < 20000 * max_step:
                            lr = (1e-4, 3e-4)
                        else:
                            lr = (3e-5, 1e-4)
                        for _ in range(episode_train_times):
                            # print('start update')
                            Q, value_loss = agent.update_policy(lr)
                            tot_Q += Q.data.cpu().numpy()
                            tot_value_loss += value_loss.data.cpu().numpy()
                        writer.add_scalar('train/critic_lr', lr[0], step)
                        writer.add_scalar('train/actor_lr', lr[1], step)
                        writer.add_scalar('train/Q', tot_Q / episode_train_times, step)
                        writer.add_scalar('train/critic_loss', tot_value_loss / episode_train_times, step)
                    if debug: prBlack('#{}: steps:{} interval_time:{:.2f} train_time:{:.2f}' \
                                      .format(episode, step, train_time_interval, time.time() - time_stamp))
                    time_stamp = time.time()
                    # reset
                    observation = None
                    episode_steps = 0
                    episode += 1

if __name__ == '__main__':
    global args
    from utils.var_args import args
    args.train_times = args.train_times*args.max_step
    args.output = get_output_folder(args.output, "Paint_edge",train_num)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)#为所有gpu设置随机种子
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    env = paint_env(args.max_step,args.env_batch,writer,width=args.width,height=args.height,train_batch=args.train_batch)
    agent = DDPG(args.train_batch, args.env_batch, args.max_step, \
                 args.tau, args.discount, args.rmsize,writer,args.resume, args.output,False)
    evaluate = Evaluator(args, writer)
    print('observation_space', env.observation_space, 'action_space', env.action_space)
    train(env,agent,evaluate)

