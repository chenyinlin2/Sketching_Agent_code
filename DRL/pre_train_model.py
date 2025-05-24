import torch

from utils.var_args import args
from DRL.actor import ResNet
from DRL.critic import ResNet_wobn
from DRL.actor_transformer import actor_transformer,init_weights
from DRL.wgan import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    '''将actor的网络参数赋值给actor_transformer'''
    path = '/data/lgf/rl_draw_image/rl_draw_edge_render_start_2_end_new_state/checkpoints/agent_model/Paint_edge-run5'
    # actor = ResNet(7 + 2, 18, 6 + 1)  # target, canvas, coordconv 3 + 3 + 2
    # actor.load_state_dict(torch.load('{}/best_actor.pkl'.format(path)))
    new_actor = actor_transformer(11, 18, 4, 1)
    init_weights(new_actor)
    new_actor.load_state_dict(torch.load('{}/best_actor.pkl'.format(path)),strict=False)
    save_path = '/data/lgf/rl_draw_image/rl_draw_edge_render_start_2_end_new_state/checkpoints/pretrain_for_model_with_transformer'
    torch.save(new_actor.state_dict(),'{}/actor.pkl'.format(save_path))

    # path = '/media/lgf/rl_draw_image/rl_draw_edge_render_change_error/checkpoints/agent_model/original_model_obs_dim_9'
    # actor = ResNet(7 + 2, 18, 6 + 1)  # target, canvas, coordconv 3 + 3 + 2
    # critic = ResNet_wobn(3 + 7 + 2, 18, 1)  # add the last canvas for better prediction
    # actor.load_state_dict(torch.load('{}/actor.pkl'.format(path)))
    # critic.load_state_dict(torch.load('{}/critic.pkl'.format(path)))
    # load_gan(path)
    #
    # new_actor = ResNet(args.obs_dim + 2, 18, args.act_dim + 1)  # target, canvas, coordconv 3 + 3 + 2
    # new_critic = ResNet_wobn(3 + args.obs_dim + 2, 18, 1)  # add the last canvas for better prediction
    #
    # for name,parms in actor.named_parameters():
    #     if 'conv1' in name or 'fc' in name:
    #         continue
    #     else:
    #         new_actor.state_dict()[name].copy_(actor.state_dict()[name])
    #         print(name)
    # print('===================================')
    # for name,parms in critic.named_parameters():
    #     if 'conv1' in name:
    #         continue
    #     else:
    #         critic.state_dict()[name].copy_(new_critic.state_dict()[name])
    #         print(name)
    # save_path = '/media/lgf/rl_draw_image/rl_draw_edge_render_change_error/checkpoints/agent_model/original_model_obs_8'
    # torch.save(new_actor.state_dict(), '{}/actor.pkl'.format(save_path))
    # torch.save(new_critic.state_dict(), '{}/critic.pkl'.format(save_path))





