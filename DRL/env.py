import cv2
import torch
from utils.util import *
from DRL.ddpg import *
from utils.var_args import args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from utils.chamfer_distance import chamfer_distance_numpy

class paint_env():
    def __init__(self,max_episode_length=10, env_batch=64,writer=None,width=240,height=240,train_batch=96):
        self.max_episode_length = max_episode_length #40
        self.env_batch = env_batch#96 环境与智能体交互的数量。每次绘制的图片是env_batch，但是保存的图片肯定会比这个多
        self.train_batch = train_batch#训练模型的时候，每次使用的batch数目
        self.obs_dim = args.obs_dim
        self.max_step = args.max_step
        self.action_space = (args.act_dim+1) #[,,,,,,]前6个是QBC曲线的坐标，第7个点是画线的颜色,第8个点时线的宽度
        self.width = width#绘制图片高度
        self.height = height#绘制图片胡
        self.observation_space = (self.env_batch, self.height, self.width, self.obs_dim)
        self.test = False
        self.writer = writer
        self.log = 0#记录writer中agent完成路径的次数
        self.first_stroke_actor = ResNet(self.obs_dim + 2, 18, args.act_dim+1)
        # self.first_stroke_actor.load_state_dict(torch.load(args.good_actor_check))
        self.first_stroke_actor.to(device)
    def reset(self,gt, test=False,first_stroke=False):
        self.test = test
        self.cur_env_batch = gt.shape[0]#env_batch是固定的，即取出来的图片数目固定，但是每张图片大小不定，会被切割。因此当前回合的图片数目不定
        self.stepnum = 0
        # stroke_start = torch.rand(self.cur_env_batch,2,dtype=torch.float)#笔画的初始位置
        # for i in range(self.cur_env_batch):
        #     gt_one = gt[i,0,...]
        #     gt_zero_idx = (255-gt_one).nonzero()
        #     if len(gt_zero_idx) != 0:
        #         gt_zero_idx_idx = random.sample(range(gt_zero_idx.shape[0]),1)[0]
        #         stroke_start[i,:] = gt_zero_idx[gt_zero_idx_idx,:]/255
        self.gt = gt.to(device)
        self.tot_reward = ((self.gt.float()/255) ** 2).mean(1).mean(1).mean(1) # [cuda]
        if self.obs_dim == 6 or self.obs_dim == 7:
            self.canvas = torch.ones([self.cur_env_batch, 3, self.height, self.width], dtype=torch.uint8).to(device)*255
        elif self.obs_dim == 2:
            self.canvas = torch.ones([self.cur_env_batch, 1, self.height, self.width], dtype=torch.uint8).to(device)*255
        else:
            raise ValueError("the input dimension of state is wrong")
        '''使用已经训练好的模型绘制第一笔，以第一笔绘制后的作为起始状态'''
        if first_stroke:
            with torch.no_grad():
                first_stroke_state = self.observation_first_stroke()
                first_stroke_inf = self.first_stroke_actor(first_stroke_state)
                self.canvas = (decode_one(first_stroke_inf,self.canvas.float()/255)*255).byte()
                self.stroke_start = (first_stroke_inf[:,4:6]*args.width).byte()
            self.stepnum += 1
        else:
            self.stroke_start = (torch.ones(self.cur_env_batch,2).to(device)).byte()
        self.lastdis = self.ini_dis = self.cal_dis()#计算gt和cancas之间的距离，即损失 [cuda]
        return self.observation() #[cuda]

    def step(self, action):
        with torch.no_grad():
            action = torch.tensor(action).to(device)
            self.stroke_start = (action[:,4:6]*args.width).byte()#取出qbc曲线的起点的坐标
            self.canvas = (decode_one(action, self.canvas.float() / 255) * 255).byte()
            self.stepnum += 1
            ob = self.observation()
            done = (self.stepnum == self.max_episode_length)
            done = np.array([done] * self.cur_env_batch)# np.array([0.] * self.batch_size)
            reward = self.cal_reward()  #(numpy)
        if done[0]:#当前路径结束时候，即走完了max_step步数
            if not self.test:
                self.dist = to_numpy(self.cal_dis())
                div_num = int(self.cur_env_batch/self.env_batch)
                for i in range(self.env_batch):
                    save_idx = i*div_num if i*div_num < self.cur_env_batch else -1
                    self.writer.add_scalar('train/dist', self.dist[save_idx], self.log)
                    self.log += 1
        return ob, reward, done, None

    def step_test(self,action):
        with torch.no_grad():
            action = torch.tensor(action).to(device)
            self.stroke_start = (action[:,4:6]*args.width).byte()#取出qbc曲线的起点的坐标
            self.canvas,stroke_with_brush,stroke = test_decode_one(action, self.canvas.float() / 255)
            self.canvas = (self.canvas*255).byte()
            self.stepnum += 1
            ob = self.observation()
            done = (self.stepnum == self.max_episode_length)
            done = np.array([done] * self.cur_env_batch)# np.array([0.] * self.batch_size)
            reward = self.cal_reward()  #(numpy)
        return ob, reward, done, self.canvas,stroke_with_brush,stroke

    def save_image(self, log, step):
        for i in range(self.env_batch):
            if i <= 10:
                canvas = to_numpy(self.canvas[i].permute(1, 2, 0))[:,:,0]
                # canvas = cv2.cvtColor((to_numpy(self.canvas[i].permute(1, 2, 0))), cv2.COLOR_BGR2GRAY)
                self.writer.add_image('{}/canvas_{}.png'.format(str(i), str(step)), canvas, log)
        if step == self.max_episode_length:
            for i in range(self.env_batch):
                if i < 50:
                    # gt = cv2.cvtColor((to_numpy(self.gt[i].permute(1, 2, 0))), cv2.COLOR_BGR2GRAY)
                    # canvas = cv2.cvtColor((to_numpy(self.canvas[i].permute(1, 2, 0))), cv2.COLOR_BGR2GRAY)
                    gt = to_numpy(self.gt[i].permute(1, 2, 0))[:, :, 0]
                    canvas = to_numpy(self.canvas[i].permute(1, 2, 0))[:, :, 0]
                    self.writer.add_image(str(i) + '/_target.png', gt, log)
                    self.writer.add_image(str(i) + '/_canvas.png', canvas, log)

    def observation(self):
        T = torch.ones([self.cur_env_batch, 1, self.height, self.width], dtype=torch.uint8) * self.stepnum
        # stroke_start = self.stroke_start.expand(self.width,self.width,self.cur_env_batch,2).permute(2,3,0,1)#[batch,2] -> [batch,2,128,128]
        state = {}
        state['state'] = torch.cat((self.canvas, self.gt, T.to(device)) ,1)
        state['start_point'] = self.stroke_start
        return state# canvas, img, T,stroke_start #[batch,1,height,width] | [batch,1,height,width] | [batch,1,height,width] | [batch,2,height,width]
        # return torch.cat((self.canvas, self.gt),1)# canvas, img, T #[batch,1,height,width] | [batch,1,height,width] | [batch,1,height,width]
    def observation_first_stroke(self):
        T = torch.ones([self.cur_env_batch, 1, self.height, self.width], dtype=torch.uint8) * self.stepnum
        return torch.cat((self.canvas, self.gt, T.to(device)/self.max_step,coord.expand(self.cur_env_batch, 2, args.height, args.width)), 1)

    def cal_dis(self):
        return (((self.canvas.float() - self.gt.float()) / 255) ** 2).mean(1).mean(1).mean(1)
    def cal_charmfer_dis(self):
        canvas_array = to_numpy(self.canvas.float()[:,0,:,:]/255)
        gt_array = to_numpy(self.gt.float()[:,0,:,:]/255)
        charmfer_dis = chamfer_distance_numpy(gt_array,canvas_array)
        return charmfer_dis
    def cal_reward(self):
        dis = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)

