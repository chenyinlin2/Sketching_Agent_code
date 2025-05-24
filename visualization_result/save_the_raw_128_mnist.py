from dataloader.mnist_loader.mnist_128_loader import getMnistTest
from utils.var_args import args
import torch
import random
import numpy as np
import os
from PIL import Image

if __name__ == '__main__':
    target_idx = 120
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.env_batch = 1
    val_loader = getMnistTest(dataset_dir='/data/DATA/MNIST/raw_128',batch_size=args.env_batch, shuffle=False)
    save_image_output_path_base = '/data/lgf/rl_draw_image/rl_draw_mnist_render_start_2_end_new_state/visualization_result/image_output'
    save_image_output_path = os.path.join(save_image_output_path_base,str(target_idx),'gt_raw.jpg')
    for i, sample_batch in enumerate(val_loader):
        if args.var_thick_flag:
            sample_batch = torch.tensor(np.array(sample_batch), dtype=torch.float)
        gt = sample_batch.to(torch.uint8)
        gt = gt.expand(gt.shape[0], 3, args.width, args.width)
        if i == target_idx:
            break
    target_img = 255 - gt.permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy()
    im = Image.fromarray(target_img)
    im.save(save_image_output_path)