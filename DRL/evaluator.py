import numpy as np
from utils.util import *
from DRL.ddpg import DDPG
from DRL.env import paint_env
from dataloader.BSDS500_loader.loaddata import getValData
from dataloader.QuickDraw_clean.var_shape_quickdraw import getQuickDrawVal

class Evaluator(object):

    def __init__(self, args, writer,shuffle=False):
        self.validate_episodes = args.validate_episodes#5
        self.max_step = args.max_step
        self.env_batch = args.env_batch
        self.var_thick_flag = args.var_thick_flag#是否绘制不同大小的图片
        self.var_patch_image = args.var_patch_image
        self.val_loader = getQuickDrawVal(batch_size=self.env_batch,img_size=128,var_thick_flag=self.var_thick_flag,var_patch_image=self.var_patch_image)
        self.writer = writer
        self.width = args.width
        self.log = 0
    def __call__(self, env:paint_env,agent_reset,policy, debug=False):
        observation = None
        dist = 0
        charmfer_dist = 0#倒角距离
        for i,sample_batch in enumerate(self.val_loader):
            if i >= self.validate_episodes:
                break
            if self.var_thick_flag:
                sample_batch = torch.tensor(np.array(sample_batch), dtype=torch.float)
            gt = sample_batch.to(torch.uint8)
            gt = gt.expand(gt.shape[0],3,self.width ,self.width)
            # if i >= self.validate_episodes:
            # reset at the start of episode
            observation = env.reset(gt)#范围是[0-255
            # agent_reset(gt)
            episode_steps = 0
            assert observation is not None            
            # start episode
            episode_reward = np.zeros(gt.shape[0])
            while (episode_steps < self.max_step or not self.max_step):
                action = policy(observation)
                observation, reward, done, (step_num) = env.step(action)
                episode_reward += reward
                episode_steps += 1
                if i == 0: env.save_image(self.log, episode_steps)
            dist += to_numpy(env.cal_dis())
            charmfer_dist += env.cal_charmfer_dis()
            self.log += 1
        dist = dist/self.validate_episodes
        charmfer_dist = charmfer_dist/self.validate_episodes
        return episode_reward, dist, charmfer_dist

    def test_actor(self,env:paint_env,agent_reset,policy):
        observation = None
        show_canvas= {}
        for i,sample_batch in enumerate(self.val_loader):
            if i >= self.validate_episodes:
                break
            if self.var_thick_flag:
                sample_batch = torch.tensor(np.array(sample_batch), dtype=torch.float)
            gt = sample_batch.to(torch.uint8)
            gt = gt.expand(gt.shape[0],3,self.width ,self.width)
            break
        observation = env.reset(gt)  # 范围是[0-255
        episode_steps = 0
        assert observation is not None
        # start episode
        canvas_list = []
        stroke_list = []
        stroke_with_brush_list = []
        while (episode_steps < self.max_step or not self.max_step):
            action = policy(observation)
            observation, reward, done, next_canvas,stroke_with_brush,stroke = env.step_test(action)
            canvas_list.append(next_canvas.permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy())
            stroke_with_brush_list.append(stroke_with_brush.permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy())
            stroke_list.append(stroke.permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy())
            episode_steps += 1
        show_canvas['canvas_list'] = canvas_list
        show_canvas['target_img'] = gt.permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy()
        show_canvas['stroke_list'] = stroke_list
        show_canvas['stroke_with_brush_list'] = stroke_with_brush_list
        return show_canvas
    '''针对单张样本，但是不限制大小，然后生成彩色图片'''
    def test_actor_with_var_thick(self,env:paint_env,agent_reset,policy,target_idx=105):
        show_canvas = {}
        for i, sample_batch in enumerate(self.val_loader):
            if i == target_idx:
                if self.var_thick_flag:
                    gt = torch.tensor(np.array(sample_batch), dtype=torch.uint8)
                else:
                    gt = sample_batch.to(torch.uint8)
                gt = gt.expand(gt.shape[0], 3, self.width, self.width)
                break
        observation = env.reset(gt)  # 范围是[0-255
        episode_steps = 0
        assert observation is not None

        canvas_list = [[] for _ in range(gt.shape[0])]  # 绘制过程中画布变化情况
        stroke_list = [[] for _ in range(gt.shape[0])]  # 绘制过程中有效笔划
        stroke_with_brush_list = [[] for _ in range(gt.shape[0])]  # 绘制过程中画笔移动轨迹
        actions_list = [[] for _ in range(gt.shape[0])]  # 绘制过程中画笔落下的笔划
        while (episode_steps < self.max_step or not self.max_step):
            action = policy(observation)  # [batch,7]
            observation, reward, done, next_canvas, stroke_with_brush, stroke = env.step_test(
                action)  # stroke_with_brush相当于画笔移动的轨迹，stroke只是画笔绘制的笔划。
            next_canvas = next_canvas.permute(0, 2, 3, 1).detach().cpu().numpy()  # [batch,128,128,3]
            stroke_with_brush = stroke_with_brush.permute(0, 2, 3, 1).detach().cpu().numpy()  # [batch,128,128,3]
            stroke = stroke.permute(0, 2, 3, 1).detach().cpu().numpy()  # [batch,128,128,1]
            episode_steps += 1
            for sample_idx in range(gt.shape[0]):
                canvas_list[sample_idx].append(next_canvas[sample_idx, ...])
                stroke_with_brush_list[sample_idx].append(stroke_with_brush[sample_idx, ...])
                stroke_list[sample_idx].append(stroke[sample_idx, ...])
                actions_list[sample_idx].append(action[sample_idx])#保存所有笔划

        show_canvas['canvas_list'] = canvas_list
        show_canvas['target_img'] = gt.permute(0,2,3,1).detach().cpu().numpy()
        show_canvas['stroke_list'] = stroke_list
        show_canvas['stroke_with_brush_list'] = stroke_with_brush_list
        return show_canvas,actions_list
    '''针对单张样本，但是不限制大小，然后生成彩色图片'''
    def test_actor_with_divide_patch_image(self,env:paint_env,policy,target_idx=105,patch_image_width=100):
        show_canvas = {}
        for i, sample_batch in enumerate(self.val_loader):
            if i == target_idx:
                if self.var_thick_flag:
                    gt = torch.tensor(np.array(sample_batch), dtype=torch.uint8)
                else:
                    gt = sample_batch.to(torch.uint8)
                gt = gt.expand(gt.shape[0], 3, self.width, self.width)
                break
        observation = env.reset(gt)  # 范围是[0-255
        episode_steps = 0
        assert observation is not None

        canvas_list = [[] for _ in range(gt.shape[0])]  # 绘制过程中画布变化情况
        stroke_list = [[] for _ in range(gt.shape[0])]  # 绘制过程中有效笔划
        stroke_with_brush_list = [[] for _ in range(gt.shape[0])]  # 绘制过程中画笔移动轨迹
        actions_list = [[] for _ in range(gt.shape[0])]  # 绘制过程中画笔落下的笔划
        while (episode_steps < self.max_step or not self.max_step):
            action = policy(observation)  # [batch,7]
            observation, reward, done, next_canvas, stroke_with_brush, stroke = env.step_test(
                action)  # stroke_with_brush相当于画笔移动的轨迹，stroke只是画笔绘制的笔划。
            next_canvas = next_canvas.permute(0, 2, 3, 1).detach().cpu().numpy()  # [batch,128,128,3]
            stroke_with_brush = stroke_with_brush.permute(0, 2, 3, 1).detach().cpu().numpy()  # [batch,128,128,3]
            stroke = stroke.permute(0, 2, 3, 1).detach().cpu().numpy()  # [batch,128,128,1]
            episode_steps += 1
            for sample_idx in range(gt.shape[0]):
                canvas_list[sample_idx].append(next_canvas[sample_idx, ...])
                stroke_with_brush_list[sample_idx].append(stroke_with_brush[sample_idx, ...])
                stroke_list[sample_idx].append(stroke[sample_idx, ...])
                actions_list[sample_idx].append(action[sample_idx])#保存所有笔划

        show_canvas['canvas_list'] = canvas_list
        show_canvas['target_img'] = gt.permute(0,2,3,1).detach().cpu().numpy()
        show_canvas['stroke_list'] = stroke_list
        show_canvas['stroke_with_brush_list'] = stroke_with_brush_list
        return show_canvas,actions_list
    '''针对多张128*128的图片，主要是保存到tensorboard中，然后进行查看。'''
    def test_actor_to_writer(self, env: paint_env, agent_reset, policy, start_idx=None, end_idx=None):
        observation = None
        color_rgb = np.array([[255, 64, 64]]) / 255  # 颜色
        if end_idx == None:
            end_idx = start_idx + 1000
        validate_num = 0
        for i, sample_batch in enumerate(self.val_loader):
            if validate_num >= start_idx and validate_num < end_idx:
                if self.var_thick_flag:
                    sample_batch = torch.tensor(np.array(sample_batch), dtype=torch.float)
                gt = sample_batch.to(torch.uint8)
                gt = gt.expand(gt.shape[0], 3, self.width, self.width)

                observation = env.reset(gt)  # 范围是[0-255
                episode_steps = 0
                assert observation is not None
                # start episode
                canvas_list = [[] for _ in range(gt.shape[0])]  # 绘制过程中画布变化情况
                stroke_list = [[] for _ in range(gt.shape[0])]  # 绘制过程中有效笔划
                stroke_with_brush_list = [[] for _ in range(gt.shape[0])]  # 绘制过程中画笔移动轨迹
                gt_with_stroke_list = [[] for _ in range(gt.shape[0])]  # 绘制过程中有效笔划和gt图的重合部分。
                gt_with_brush_list = [[] for _ in range(gt.shape[0])]  # 绘制过程中画笔移动轨迹和gt图的重合部分

                target_img = gt.permute(0, 2, 3, 1).detach().cpu().numpy()
                while (episode_steps < self.max_step or not self.max_step):
                    action = policy(observation)  # [batch,7]
                    observation, reward, done, next_canvas, stroke_with_brush, stroke = env.step_test(
                        action)  # stroke_with_brush相当于画笔移动的轨迹，stroke只是画笔绘制的笔划。
                    next_canvas = next_canvas.permute(0, 2, 3, 1).detach().cpu().numpy()  # [batch,128,128,3]
                    stroke_with_brush = stroke_with_brush.permute(0, 2, 3,
                                                                  1).detach().cpu().numpy()  # [batch,128,128,3]
                    stroke = stroke.permute(0, 2, 3, 1).detach().cpu().numpy()  # [batch,128,128,1]
                    gt_with_stroke = target_img * (1 - stroke) / 255 + stroke * color_rgb.reshape(1, 1,
                                                                                                  3)  # [batch,128,128,3]
                    gt_with_brush = target_img * (1 - stroke_with_brush) / 255 + stroke_with_brush * color_rgb.reshape(
                        1, 1, 3)  # [batch,128,128,3]
                    for sample_idx in range(gt.shape[0]):
                        canvas_list[sample_idx].append(next_canvas[sample_idx, ...])
                        stroke_with_brush_list[sample_idx].append(stroke_with_brush[sample_idx, ...])
                        stroke_list[sample_idx].append(stroke[sample_idx, ...])
                        gt_with_stroke_list[sample_idx].append(gt_with_stroke[sample_idx, ...])
                        gt_with_brush_list[sample_idx].append(gt_with_brush[sample_idx, ...])
                    episode_steps += 1
                for sample_idx in range(gt.shape[0]):
                    canvas_tensor = torch.tensor(np.array(canvas_list[sample_idx]))
                    stroke_tensor = torch.tensor(np.array(stroke_list[sample_idx]))
                    stroke_with_brush_tensor = torch.tensor(np.array(stroke_with_brush_list[sample_idx]))
                    gt_with_stroke_tensor = torch.tensor(np.array(gt_with_stroke_list[sample_idx]))
                    gt_with_brush_tensor = torch.tensor(np.array(gt_with_brush_list[sample_idx]))
                    writer_gt_and_final = torch.cat([gt[sample_idx:sample_idx + 1, ],
                                                     torch.tensor(next_canvas[sample_idx:sample_idx + 1]).permute(0, 3,
                                                                                                                  1,
                                                                                                                  2)],
                                                    dim=0)
                    self.writer.add_images(tag='{}/1_target_and_final'.format(validate_num + sample_idx),
                                           img_tensor=writer_gt_and_final)
                    self.writer.add_images(tag='{}/2_canvas_list'.format(validate_num + sample_idx),
                                           img_tensor=canvas_tensor, dataformats='NHWC')
                    self.writer.add_images(tag='{}/3_gt_with_stroke_list'.format(validate_num + sample_idx),
                                           img_tensor=gt_with_stroke_tensor, dataformats='NHWC')
                    self.writer.add_images(tag='{}/4_stroke_list'.format(validate_num + sample_idx),
                                           img_tensor=stroke_tensor, dataformats='NHWC')
                    self.writer.add_images(tag='{}/5_gt_with_brush_list'.format(validate_num + sample_idx),
                                           img_tensor=gt_with_brush_tensor, dataformats='NHWC')
                    self.writer.add_images(tag='{}/6_stroke_with_brush_list'.format(validate_num + sample_idx),
                                           img_tensor=stroke_with_brush_tensor, dataformats='NHWC')
                    print('writer is ok')
                print(str(validate_num) + '->' + str(validate_num + gt.shape[0]) + ' is validated')
            elif validate_num >= end_idx:
                break
            validate_num += sample_batch.shape[0]
        return True
