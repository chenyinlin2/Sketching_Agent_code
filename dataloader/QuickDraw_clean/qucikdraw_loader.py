import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import random
from dataloader.QuickDraw_clean.RealRenderer import GizehRasterizor as RealRenderer
class QuickDraw(Dataset):
    def __init__(self,dataset_base_dir,datakind='train',line_thickness=2,img_size=128):
        self.line_thickness = line_thickness#绘制线的宽度
        self.img_size = img_size#绘制的图片的大小
        self.is_bin = True
        self.rasterizor = RealRenderer()
        self.stroke3_list = load_dataset_multi_object(dataset_base_dir,datakind)
        if datakind == 'test':
            random.shuffle(self.stroke3_list)
        # pass
    def __getitem__(self, idx):
        stroke3_data = self.stroke3_list[idx]
        stroke_image = self.get_stroke_to_image(stroke3_data)
        return stroke_image
    def __len__(self):
        return len(self.stroke3_list)

    def get_stroke_to_image(self,stroke3_data):
        # canvas = np.zeros(shape=(self.img_size,self.img_size), dtype=np.float32)
        gt_image_array = self.rasterizor.raster_func([stroke3_data], self.img_size, stroke_width=self.line_thickness,
                                                     is_bin=self.is_bin, version='v2')#list[1] : [width,width]
        gt_image_array = np.stack(gt_image_array, axis=0)#[1,width,width] 黑底白笔画 [0.0-BG, 1.0-strokes]
        # gt_image_array = 1.0 - gt_image_array  # (batch_size, image_size, image_size), [0.0-strokes, 1.0-BG]白底黑笔画
        # stroke_image = 1.0 - gt_image_array[0]  # (image_size, image_size), [0.0-BG, 1.0-strokes]
        return (1-gt_image_array)*255

def getQuickDrawTrain(dataset_dir='/data/DATA/QuickDraw-clean',batch_size=48,img_size=128,line_thickness=3):#路径 | batch | 图像大小 | 线段宽度
    data_kind = 'train'
    dataset = QuickDraw(dataset_dir,data_kind,img_size=img_size,line_thickness=line_thickness)
    dataloader_training = DataLoader(dataset, batch_size,
                                     shuffle=True, num_workers=16, pin_memory=True,drop_last=True)
    return dataloader_training

def getQuickDrawVal(dataset_dir='/data/DATA/QuickDraw-clean',batch_size=1,img_size=128,line_thickness=3):#路径 | batch | 图像大小 | 线段宽度
    data_kind = 'test'
    dataset = QuickDraw(dataset_dir,data_kind,img_size=img_size,line_thickness=line_thickness)
    dataloader_testing = DataLoader(dataset, batch_size,
                                     shuffle=False, num_workers=1, pin_memory=False,drop_last=False)
    return dataloader_testing

def load_qd_npz_data(npz_path):
    data = np.load(npz_path, encoding='latin1', allow_pickle=True)
    selected_strokes3 = data['stroke3']  # (N_sketches,), each with (N_points, 3)
    selected_strokes3 = selected_strokes3.tolist()  # [5000,29,3] -> list[5000]:[29,3]
    return selected_strokes3
def load_dataset_multi_object(dataset_base_dir,datakind='train'):
    train_stroke3_data = []#记录训练集 list[50000]([29,3])
    # val_stroke3_data = []#记录测试集 list[5000]([29,3])
    #选择加载的类别
    cates = ['airplane', 'bus', 'car', 'sailboat', 'bird', 'cat', 'dog',
             # 'rabbit',
             'tree', 'flower',
             # 'circle', 'line',
             'zigzag'
             ]
    for cate in cates:
        train_cate_sketch_data_npz_path = os.path.join(dataset_base_dir,  datakind, cate + '.npz')
        train_cate_stroke3_data = load_qd_npz_data(train_cate_sketch_data_npz_path)  # list of (N_sketches,), each with (N_points, 3)
        train_stroke3_data += train_cate_stroke3_data
        print(train_cate_sketch_data_npz_path)
    print('Loaded {} from {}'.format(len(train_stroke3_data), datakind))
    return train_stroke3_data
if __name__ == '__main__':
    import imageio
    quickdraw_dir = '/data/DATA/QuickDraw-clean'
    dataloader_training = getQuickDrawTrain(quickdraw_dir,batch_size=96)
    import time
    import torch
    device = torch.device('cuda')
    start_time = time.time()
    for i,sample_batch in enumerate(dataloader_training):
        stroke_batch = sample_batch
        # stroke_tensor = stroke_batch.to(device)
        # imageio.imwrite('./stroke_{}.png'.format(i), sample_batch[0,0,:,:])
        if i % 10 == 0:
            show_array = np.array(sample_batch[0, 0, :, :])
            plt.imshow(sample_batch[0, 0, :, :], cmap='gray')
            # plt.savefig('./stroke_{}.png'.format(i))
            plt.show()
        print(time.time() - start_time)
        start_time = time.time()
        if i > 100:
            break
    # load_dataset_multi_object(quickdraw_dir)
