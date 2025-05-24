import argparse

parser = argparse.ArgumentParser(description='Learning to Paint edge')
# hyper-parameter
parser.add_argument('--warmup', default=100, type=int,
                    help='timestep without training but only filling the replay memory')  # 训练前先保存到memory中的个数
parser.add_argument('--discount', default=0.9, type=float, help='discount factor')  # 折扣因子
# parser.add_argument('--batch_size', default=96, type=int, help='minibatch size')
parser.add_argument('--rmsize', default=1000, type=int, help='replay memory size')  # memory size
parser.add_argument('--env_batch', default=96, type=int, help='concurrent environment number')
parser.add_argument('--train_batch', default=96, type=int, help='train model use number of img')
parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')  # 软更新
parser.add_argument('--max_step', default=40, type=int, help='max length for episode')  # 每个回合采取的最大步数
parser.add_argument('--noise_factor', default=0.6, type=float, help='noise level for parameter space noise')  # 是否有噪声探索空间
parser.add_argument('--validate_interval', default=50, type=int,
                    help='how many episodes to perform a validation')  # 训练50次后进行一次validate
parser.add_argument('--validate_episodes', default=5, type=int,
                    help='how many episode to perform during validation')  # 在validate中测试次数
parser.add_argument('--train_times', default=50000, type=int, help='total traintimes')  #
parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')  # 每个数据用来更新的次数。
parser.add_argument('--resume', default="/data/lgf/rl_draw_image/rl_draw_edge_render_start_2_end_new_state/checkpoints/pretrain_for_model_with_transformer", type=str, help='Resuming model path for testing')
parser.add_argument('--output', default='./checkpoints/agent_model', type=str, help='Resuming model path for testing')
# parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
parser.add_argument('--debug', default=True, type=bool, help='print some info')
parser.add_argument('--seed', default=1234, type=int, help='random seed')
parser.add_argument('--width', default=128, type=int, help='image width')
parser.add_argument('--height', default=128, type=int, help='image height')
'''==========testing various==========='''
parser.add_argument('--actor_check', default='/data/lgf/rl_draw_image/rl_draw_edge_render_start_2_end_new_state/checkpoints/agent_model/Paint_edge-run1/actor.pkl', type=str, help='the checkpoints of actor model')
# parser.add_argument('--good_actor_check', default='/data/lgf/rl_draw_image/rl_draw_edge_render_start_2_end_new_state/checkpoints/agent_model/Paint_edge-run1/best_actor.pkl', type=str, help='the checkpoints of actor model')
# parser.add_argument('--actor_check', default='/data/lgf/rl_draw_image/rl_draw_edge_render_start_2_end/checkpoints/agent_model/Paint_edge-run1/actor.pkl', type=str, help='the checkpoints of actor model')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
parser.add_argument('--act_dim', default=6, type=int, help='the act dim')#绘制一条线段用的参数，不包括颜色
parser.add_argument('--obs_dim', default=7, type=int, help='the obs dim')#绘制一条线段用的参数，不包括颜色
parser.add_argument('--divide', default=2, type=int, help='divide the target image to get better resolution')
parser.add_argument('--var_thick_flag', default=False, type=bool, help='use different stroke to draw different width img')
parser.add_argument('--var_patch_image', default=False, type=bool, help='different width patch img')
args = parser.parse_args()
args.good_actor_check = args.actor_check.replace('actor.pkl','best_actor.pkl')
# args.best_actor_check = args.actor_check.replace('actor.pkl','best_actor.pkl')



