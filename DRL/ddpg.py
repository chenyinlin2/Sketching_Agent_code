import matplotlib.pyplot as plt
from Renderer.model import FCN,FCN_256
import torch
from Renderer.stroke_gen import draw_circle_2
from DRL.actor import ResNet
from DRL.critic import ResNet_wobn
from DRL.actor_detr_transformer import actor_transformer
from DRL.vit_transformer import actor_Vit
from torch.optim import Adam, SGD
from utils.var_args import args
from utils.util import *
from DRL.rpm import *
from DRL.wgan import *
from Renderer.class_model import Class_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# height,width = 256,256
coord = torch.zeros([1, 2, args.height, args.width])
for i in range(args.height):
    for j in range(args.width):
        coord[0, 0, i, j] = i / 255.
        coord[0, 1, i, j] = j / 255.
coord = coord.to(device)
stork_num = 1#每次动作中笔画的数目
color_expand = torch.ones([1, 3], dtype=torch.float).to(device)

# Decoder = FCN_256()
# Decoder.load_state_dict(torch.load('../checkpoints/renderer_model/renderer_256.pth'))
Decoder = FCN(input_dim=args.act_dim)#get [B:1,]
Decoder.load_state_dict(torch.load('/data/lgf/rl_draw_image/rl_draw_edge_render_start_2_end_new_state/checkpoints/renderer_model/render_3/renderer_499999.pth'))
# Decoder.load_state_dict(torch.load('./checkpoints/renderer_model/render_5/renderer_499999.pth'))
color_class = Class_model()
color_class.load_state_dict(torch.load('/data/lgf/rl_draw_image/rl_draw_edge_render_start_2_end_new_state/checkpoints/class_model/train_time_2/class_20000.pth'))
if torch.cuda.is_available():
    Decoder = Decoder.cuda()
    color_class.cuda()
def decode(x, canvas): # b * (8)
    x = x.view(-1, args.act_dim + 1)#[1,13]
    stroke = 1 - Decoder(x[:, :args.act_dim])#[1,128,128] [B:1,S:0] -> [B:0,S:1]
    stroke = stroke.view(-1, args.width, args.width, 1)#[1,128,128] -> [1,128,128,1]
    color_expand_ = color_expand.expand(x.shape[0], 3)
    color_stroke = stroke * (color_class(x[:, -1:])*color_expand_).view(-1, 1, 1, 3) # [stork_num,128,128,1]*[stork_num,1,1,1] -> [stork_num,128,128,1]
    stroke = stroke.permute(0, 3, 1, 2)#[5,128,128,1] -> [5,1,128,128]
    color_stroke = color_stroke.permute(0, 3, 1, 2)#[5,128,128,1] -> [5,1,128,128]
    stroke = stroke.view(-1, stork_num, 1, args.width, args.width)#[5,1,128,128] -> [1,5,1,128,128]
    color_stroke = color_stroke.view(-1, stork_num, 3, args.width, args.width)#[5b,3,128,128] -> [b,5,3,128,128]
    for i in range(stork_num):
        canvas = canvas * (1-stroke[:, i]) + color_stroke[:,i]#(1 - stroke[:, i]) + color_stroke[:, i]
        # canvas = canvas * (1-stroke[:, i]) #+ color_stroke[:,i]#(1 - stroke[:, i]) + color_stroke[:, i]
    return canvas

def decode_one(x, canvas): # b * (8)
    x = x.view(-1, args.act_dim + 1)#[1,13]
    stroke = 1 - Decoder(x[:, :args.act_dim])#[1,128,128] [B:1,S:0] -> [B:0,S:1]
    stroke = stroke.view(-1, args.width, args.width, 1)#[1,128,128] -> [1,128,128,1]
    color_expand_ = color_expand.expand(x.shape[0], 3)
    color_stroke = stroke * (color_class(x[:, -1:])*color_expand_).view(-1, 1, 1, 3) # [stork_num,128,128,1]*[stork_num,1,1,1] -> [stork_num,128,128,1]
    stroke = stroke.permute(0, 3, 1, 2)#[5,128,128,1] -> [5,1,128,128]
    color_stroke = color_stroke.permute(0, 3, 1, 2)#[5,128,128,1] -> [5,1,128,128]
    stroke = stroke.view(-1, stork_num, 1, args.width, args.width)#[5,1,128,128] -> [1,5,1,128,128]
    color_stroke = color_stroke.view(-1, stork_num, 3, args.width, args.width)#[5b,3,128,128] -> [b,5,3,128,128]
    for i in range(stork_num):
        # canvas = canvas * (1-stroke[:, i]) + color_stroke[:,i]#(1 - stroke[:, i]) + color_stroke[:, i]
        canvas = canvas * (1-color_stroke[:,i]) #(1 - stroke[:, i]) + color_stroke[:, i]
        # canvas = canvas * (1-stroke[:, i]) #+ color_stroke[:,i]#(1 - stroke[:, i]) + color_stroke[:, i]
    return canvas
'''采用canvas*（1-stroke）+color_stroke的方式，代表1是绘制白色，0是绘制黑色。即能够绘制两种颜色。那么采用的color_class模型就应该在'''
'''采用canvas*(1-color_stroke)的方式，这时0和1分别表示画笔是否落下，如果输出1，代表落下，则绘制出来黑色，反之0代表不落下，则不会绘制'''
def test_decode_one(x, canvas): # b * (8)
    x = x.view(-1, args.act_dim + 1)#[1,13]
    stroke = 1 - Decoder(x[:, :args.act_dim])#[1,128,128] [B:1,S:0] -> [B:0,S:1]
    stroke = stroke.view(-1, args.width, args.width, 1)#[1,128,128] -> [1,128,128,1]
    color_expand_ = color_expand.expand(x.shape[0], 3)
    color_stroke = stroke * (color_class(x[:, -1:])*color_expand_).view(-1, 1, 1, 3) # [stork_num,128,128,1]*[stork_num,1,1,1] -> [stork_num,128,128,1]
    stroke = stroke.permute(0, 3, 1, 2)#[5,128,128,1] -> [5,1,128,128]
    color_stroke = color_stroke.permute(0, 3, 1, 2)#[5,128,128,1] -> [5,1,128,128]
    stroke = stroke.view(-1, stork_num, 1, args.width, args.width)#[5,1,128,128] -> [1,5,1,128,128]
    color_stroke = color_stroke.view(-1, stork_num, 3, args.width, args.width)#[5b,3,128,128] -> [b,5,3,128,128]
    for i in range(stork_num):
        # canvas = canvas * (1-stroke[:, i]) + color_stroke[:,i]#(1 - stroke[:, i]) + color_stroke[:, i]
        canvas = canvas * (1-color_stroke[:,i]) #(1 - stroke[:, i]) + color_stroke[:, i]
        # canvas = canvas * (1-stroke[:, i]) #+ color_stroke[:,i]#(1 - stroke[:, i]) + color_stroke[:, i]
    return canvas,(stroke*color_stroke)[:,0,...],stroke[:,0,...]
class DDPG(object):
    def __init__(self, train_batch=64, env_batch=1, max_step=40, \
                 tau=0.001, discount=0.9, rmsize=800, \
                 writer=None, resume=None, output_path=None,only_gan=True,):
        self.max_step = max_step
        self.env_batch = env_batch
        self.train_batch = train_batch
        self.obs_dim = args.obs_dim
        self.actor = ResNet(self.obs_dim + 2 + 2, 18, args.act_dim-2+1)# target, canvas, stepnum,coordconv 3 + 3 + 2
        self.actor_target = ResNet(self.obs_dim + 2 + 2, 18, args.act_dim-2+1)#state: [canvas,target,stepnum,stroke_start]
        # self.actor = actor_Vit(self.obs_dim+2+2,args.act_dim-2,1)# target, canvas, stepnum,coordconv 3 + 3 + 2
        # self.actor_target = actor_Vit(self.obs_dim + 2 + 2,args.act_dim-2,1)#state: [canvas,target,stepnum,stroke_start]

        if self.obs_dim == 6 or self.obs_dim == 7:
            self.critic = ResNet_wobn(3 + self.obs_dim + 2 + 2, 18, 1)# add the last canvas for better prediction [next_canvas,target, canvas, stepnum,coordconv,curse_start]
            self.critic_target = ResNet_wobn(3 + self.obs_dim + 2 + 2, 18, 1)# add the last canvas for better prediction
        elif self.obs_dim == 2:
            self.critic_target = ResNet_wobn(1 + self.obs_dim + 2, 18, 1)
            self.critic = ResNet_wobn(1 + self.obs_dim + 2, 18, 1)
        else:
            raise ValueError('the input dim is error')
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-2)
        self.criterion = nn.MSELoss()

        if (resume != None):#加载预训练参数
            self.only_gan = only_gan
            if self.only_gan:
                load_gan(resume,kind='best')
                # self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(resume)))
                print('only load gan model successfully')
            else:
                self.load_weights(resume,)
                print('load pretrained model successfully')
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        # Create replay buffer
        self.memory = rpm(rmsize * max_step)

        # Hyper-parameters
        self.tau = tau
        self.discount = discount

        # Tensorboard
        self.writer = writer
        self.log = 0

        self.state = {'state':[None] * self.env_batch,'start_point':[None] * self.env_batch}  # Most recent state
        self.action = [None] * self.env_batch  # Most recent action
        self.choose_device()
    def play(self, state,target=False):#[canvas,target,stepnum,stroke_start]
        # state = torch.cat((state.float() / 255,  coord.expand(state.shape[0], 2, args.height, args.width)), 1)
        state_sub = state['state']#[96,7,128,128]
        start_point = state['start_point'].float()/args.width#[96,2]
        start_point.requires_grad = True
        start_point_canvas = start_point.view(state_sub.shape[0],2,1,1).expand(state_sub.shape[0],2,args.width,args.width)#[96,2] -> [96,2,128,128]

        state_all = torch.cat((start_point_canvas,state_sub[:,0:6].float() / 255, state_sub[:, 6:7].float()/self.max_step,coord.expand(state_sub.shape[0], 2, args.height, args.width)), 1)
        if target:
            action_sub = self.actor_target(state_all)
        else:
            action_sub = self.actor(state_all)

        action_start = start_point.detach()
        # b_1 = torch.zeros(action_sub.shape[0], 2, 1, args.width).to(device)
        # b_1[:, :, 0, 0] = 1
        # b_2 = torch.zeros(action_sub.shape[0], 2, args.width, 1).to(device)
        # b_2[:, :, 0, 0] = 1
        # action_start = torch.matmul(torch.matmul(b_1,state_input), b_2).view(-1,2)

        # action_start = state[:,7:9,0,0].detach()
        action = torch.cat((action_start,action_sub),dim=1)#动作需要加上最开始的
        return action

    def noise_action(self, noise_factor, state, action):
        # noise = np.zeros(action.shape)
        for i in range(self.cur_env_batch):
            action[i,-1] = action[i,-1] + np.random.normal(0, self.noise_level[i], 1).astype('float32')
        return np.clip(action.astype('float32'), -1, 1)

    #根据状态确定动作
    def select_action(self, state, return_fix=False, noise_factor=0):
        self.eval()
        with torch.no_grad():
            action = self.play(state)#[]
            action = to_numpy(action)#tensor -> ndarray
        if noise_factor > 0:
            action = self.noise_action(noise_factor, state, action)
        self.train()
        self.action = action#cpu
        if return_fix:
            return action
        return self.action

    def reset(self, obs, factor,step):
        self.state = obs#[canvas,target,stepnum,stroke_start]
        self.cur_step = step#当前回合的起始步。
        self.cur_env_batch = obs['state'].shape[0]#current env_batch is divided to cur_env_batch
        if factor > 0:
            if step < args.train_times/4:factor = factor
            elif step < args.train_times/2:factor = factor/4
            elif step < args.train_times:factor = factor/16
        self.noise_level = np.random.uniform(0, factor, self.cur_env_batch)

    def observe(self, reward, state, done, step):
        # s0 = torch.tensor(self.state, device='cpu')
        s0_state = self.state['state'].detach().cpu()
        s0_start_point = self.state['start_point'].detach().cpu()
        a = to_tensor(self.action, "cpu")
        r = to_tensor(reward, "cpu")
        # s1 = torch.tensor(state, device='cpu')
        s1_state = state['state'].detach().cpu()
        s1_start_point = state['start_point'].detach().cpu()
        d = to_tensor(done.astype('float32'), "cpu")
        for i in range(self.cur_env_batch):
            self.memory.append([s0_state[i],s0_start_point[i], a[i], r[i], s1_state[i],s1_start_point[i],d[i]])
        self.state = state
    def evaluate(self, state, action, target=False):
        state_sub = state['state']
        T = state_sub[:, 6:7]#[96,1,128,128]
        gt = state_sub[:, 3:6].float() / 255##[96,3,128,128]
        canvas0 = state_sub[:, 0:3].float() / 255  # [96,3,128,128]
        canvas1 = decode_one(action, canvas0)#[96,3,128,128]

        # next_stroke_input = action[:,4:6].view(-1,2,1,1)
        # next_stroke_input_expand = torch.ones(action.shape[0],2,args.width,args.width).to(device)
        # next_stroke_start = next_stroke_input * next_stroke_input_expand
        '''将笔划的结束点与gt中对应位置点的像素值关联起来，做到可以微分'''
        stroke_end_point = action[:,4:6].view(self.train_batch,2,1).expand(action.shape[0],2,3)
        stroke_end_point = stroke_end_point.permute(0,2,1).reshape(action.shape[0],-1)
        stroke_end_point_canvas = 1-Decoder(stroke_end_point)#[B:1,S:0] -> [B:0,S:1]
        stroke_end_point_canvas = stroke_end_point_canvas.view(self.train_batch, args.width, args.width, 1).expand(self.train_batch,  args.width,  args.width, 3).permute(0, 3,
                                                                                                                1, 2)
        stroke_end_point_pixel = 800*((0.5-gt)*stroke_end_point_canvas).view(self.train_batch,-1).mean(dim=1,keepdims=True)
        if self.cur_step > self.max_step * 5000:
            gan_reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt) + stroke_end_point_pixel + 0.2*(0.5-color_class(action[:, -1:]))#*(1-color_class(action[:, -1:])) #+ 0.5 * color_class(action[:, -1:]) #+ #(96, 64)
        else:
            gan_reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt) + stroke_end_point_pixel
        # if T[:,0,0,0] < args.max_step//3:

        # L2_reward = torch.unsqueeze(((canvas0 - gt) ** 2).mean(1).mean(1).mean(1) - ((canvas1 - gt) ** 2).mean(1).mean(1).mean(1),1)
        coord_ = coord.expand(state_sub.shape[0], 2, args.height, args.width)#[96,2,128,128]

        stroke_end_point_state = action[:,4:6]
        stroke_end_point_state = stroke_end_point_state.view(self.train_batch,2,1,1).expand(self.train_batch,2,args.width,args.width)
        merged_state = torch.cat([canvas0,canvas1,gt,stroke_end_point_state,(T + 1).float() / self.max_step,coord_], 1)##[96,5,128,128]
        if target:
            Q = self.critic_target(merged_state)#[96,64]
            return (Q+gan_reward),gan_reward
        else:
            Q = self.critic(merged_state)
            if self.log % 20 == 0:
                self.writer.add_scalar('train/expect_reward', Q.mean(), self.log)
                self.writer.add_scalar('train/gan_reward', gan_reward.mean(), self.log)
                self.writer.add_scalar('train/stroke_end_reward', stroke_end_point_pixel.mean(), self.log)
            return (Q+gan_reward), gan_reward

    def update_policy(self,lr):
        self.log += 1
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr[1]
        # Sample batch
        state_sub,state_start_point, action, reward, \
            next_state_sub,next_state_start_point, terminal = self.memory.sample_batch(self.train_batch, device,item_count_=7)
        state = {'state':state_sub,'start_point':state_start_point}
        next_state = {'state':next_state_sub,'start_point':next_state_start_point}
        if self.log%40 == 0:
            self.writer.add_scalar('train/dist_reward', reward.mean(), self.log)
        #[96,3,128,128] | [96,7] | [96] | [96,3,128,128] | [96] ( all tensor)
        # self.update_gan(next_state)
        if self.only_gan:#如果只让gan网络读取预训练参数，则gan网络更新滞后。
            if self.cur_step > self.max_step * 8000: self.update_gan(next_state)
        else:
            if self.cur_step > self.max_step * 8000: self.update_gan(next_state)

        with torch.no_grad():
            next_action = self.play(next_state, True)#[96,7]
            target_q, _ = self.evaluate(next_state, next_action, True)#[96,3,128,128] [96,7] -> [96,64],[96,64]
            target_q = self.discount * ((1 - terminal.float()).view(-1, 1)) * target_q

        cur_q, step_reward = self.evaluate(state, action)#[96,3,128,128],[96,9]
        target_q += step_reward.detach()

        value_loss = self.criterion(cur_q, target_q)
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()

        action = self.play(state)
        pre_q, _ = self.evaluate({'state':state['state'].detach(),'start_point':state['start_point'].detach()}, action)
        policy_loss = -pre_q.mean()
        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        # exit()
        return -policy_loss, value_loss

    def update_gan(self, state):
        state_sub = state['state']
        canvas = state_sub[:, :3]
        gt = state_sub[:, 3 : 6]
        fake, real, penal = update(canvas.float() / 255, gt.float() / 255)
        if self.log % 20 == 0:
            self.writer.add_scalar('train/gan_fake', fake, self.log)
            self.writer.add_scalar('train/gan_real', real, self.log)
            self.writer.add_scalar('train/gan_penal', penal, self.log)

    def load_weights(self, path,kind=None):
        if kind==None:
            self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(path)),strict=False)
            self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(path)),strict=False)
            load_gan(path)
        elif kind=='best':
            self.actor.load_state_dict(torch.load('{}/best_actor.pkl'.format(path)),strict=False)
            self.critic.load_state_dict(torch.load('{}/best_critic.pkl'.format(path)),strict=False)
            load_gan(path,kind)
    def save_model(self, path,kind=None):
        self.actor.cpu()
        self.critic.cpu()
        if kind == None:
            torch.save(self.actor.state_dict(),'{}/actor.pkl'.format(path))
            torch.save(self.critic.state_dict(),'{}/critic.pkl'.format(path))
            save_gan(path)
        else:
            torch.save(self.actor.state_dict(),'{}/{}_actor.pkl'.format(path,kind))
            torch.save(self.critic.state_dict(),'{}/{}_critic.pkl'.format(path,kind))
            save_gan(path,kind)
        self.choose_device()
    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()
    def choose_device(self):
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)




