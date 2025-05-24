import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 确定使用哪一块gpu
import random
from dataloader.QuickDraw_clean.RealRenderer import GizehRasterizor as RealRenderer
class QuickDraw(Dataset):
    def __init__(self,dataset_base_dir,datakind='train',line_thickness=2,img_size=128,var_thick_flag=False,var_patch_image=False):
        self.img_size_set = np.array([256])
        self.var_thick_flag = var_thick_flag #是否有不同宽度的图片的标志位
        # self.line_thickness_set = self.img_size_set/
        self.line_thickness = line_thickness#绘制线的宽度
        self.img_size = img_size#绘制的图片的大小
        self.is_bin = True
        self.rasterizor = RealRenderer()
        self.stroke3_list = load_dataset_multi_object(dataset_base_dir,datakind)
        self.data_kind = datakind#目前是训练还是测试的标志。
        self.var_patch_image = var_patch_image
        if datakind == 'test':
            random.shuffle(self.stroke3_list)
        # pass
    def __getitem__(self, idx):
        stroke3_data = self.stroke3_list[idx]
        if self.var_thick_flag:
            if self.data_kind == 'test':
                random_img_size = self.img_size_set[idx % len(self.img_size_set)]#根据idx设置图片的大小
                # print('random_img_size',random_img_size)
                # line_thickness_set = np.arange(3,3*random_img_size/128+0.1,2).astype(int)#根据图片的大小，随机设置生成笔划应该有的宽度的取值范围。
                line_thickness_set = np.arange(3,3+0.1,2).astype(int)#根据图片的大小，随机设置生成笔划应该有的宽度的取值范围。
                random_line_thickness = line_thickness_set[idx % len(line_thickness_set)]#根据图片的大小，随机设置生成笔划应该有的宽度。
                # print('random_line_thickness',random_line_thickness)
                stroke_image = self.get_stroke_to_image(stroke3_data,self.var_thick_flag,random_img_size,random_line_thickness)#[d*d,1,128,128]
            elif self.data_kind == 'train':
                random_img_size = np.random.choice(self.img_size_set)#随机设置图片的大小
                # line_thickness_set = np.arange(3,3*random_img_size/128+0.1,2).astype(int)#根据图片的大小，随机设置生成笔划应该有的宽度的取值范围。
                line_thickness_set = np.arange(3,3+0.1,2).astype(int)#根据图片的大小，随机设置生成笔划应该有的宽度的取值范围。
                random_line_thickness = np.random.choice(line_thickness_set)#根据图片的大小，随机设置生成笔划应该有的宽度。
                stroke_image = self.get_stroke_to_image(stroke3_data,self.var_thick_flag,random_img_size,random_line_thickness)#[d*d,1,128,128]
            else:
                raise ValueError('Wrong kind,choose from test or train')
        else:
            stroke_image = self.get_stroke_to_image(stroke3_data,self.var_thick_flag)#[1,128,128]

        return stroke_image
    def __len__(self):
        return len(self.stroke3_list)

    def get_stroke_to_image(self,stroke3_data,var_thick_flag=False,random_img_size=None,random_line_thickness=None,var_patch_image=False):
        # canvas = np.zeros(shape=(self.img_size,self.img_size), dtype=np.float32)
        if not var_thick_flag:
            gt_image_array = self.rasterizor.raster_func([stroke3_data], self.img_size, stroke_width=self.line_thickness,
                                                         is_bin=self.is_bin, version='v2')#list[1] : [width,width]
            gt_image_array = np.stack(gt_image_array, axis=0)#[1,width,width] 黑底白笔画 [0.0-BG, 1.0-strokes]
            return (1 - gt_image_array) * 255
        else:#应该生成不同宽度的笔划绘制的不同大小的图片
            divide = int(random_img_size/self.img_size)#每列应该分成的图片的大小
            gt_image_array = self.rasterizor.raster_func([stroke3_data], random_img_size, stroke_width=random_line_thickness,
                                                         is_bin=self.is_bin, version='v2')#list[1] : [width,width]
            gt_image_array = np.stack(gt_image_array, axis=0).squeeze()#[1,width,width] -> [width,width] 黑底白笔画 [0.0-BG, 1.0-strokes]
            gt_image_array = gt_image_array.reshape((divide,self.img_size,divide,self.img_size,-1))#[width,width] ->[d,128,d,128]
            gt_image_array = np.transpose(gt_image_array, (0, 2, 1, 3, 4))#[4,128,4,128,1] -> [4,4,128,128,1]
            gt_image_array = gt_image_array.reshape((divide ** 2, self.img_size, self.img_size, -1))#[d,d,128,128,1] -> [d*d,128,128,1]
            gt_image_array = np.transpose(gt_image_array, (0, 3, 1, 2))#[d*d,128,128,1] -> [d*d,1,128,128]
            return (1 - gt_image_array) * 255
        # gt_image_array = 1.0 - gt_image_array  # (batch_size, image_size, image_size), [0.0-strokes, 1.0-BG]白底黑笔画
        # stroke_image = 1.0 - gt_image_array[0]  # (image_size, image_size), [0.0-BG, 1.0-strokes]

    def mine_collate_fn(self,batch):
        batch_list = [c for b in batch for c in b]
        return batch_list

def getQuickDrawTrain(dataset_dir='/data/DATA/QuickDraw-clean',batch_size=48,img_size=128,line_thickness=3,var_thick_flag=False):#路径 | batch | 图像大小 | 线段宽度
    data_kind = 'train'
    dataset = QuickDraw(dataset_dir,data_kind,img_size=img_size,line_thickness=line_thickness,var_thick_flag=var_thick_flag)
    if var_thick_flag:#需要使用不同宽度的笔划绘制出来不同大小的图片
        dataloader_training = DataLoader(dataset, batch_size,
                                         shuffle=True, num_workers=16, pin_memory=True,drop_last=True,collate_fn=dataset.mine_collate_fn)
    else:
        dataloader_training = DataLoader(dataset, batch_size,
                                         shuffle=True, num_workers=16, pin_memory=True,drop_last=True)
    return dataloader_training

def getQuickDrawVal(dataset_dir='/data/DATA/QuickDraw-clean',batch_size=1,img_size=128,line_thickness=3,var_thick_flag=False,var_patch_image=False):#路径 | batch | 图像大小 | 线段宽度
    data_kind = 'test'
    dataset = QuickDraw(dataset_dir,data_kind,img_size=img_size,line_thickness=line_thickness,var_thick_flag=var_thick_flag,var_patch_image=var_patch_image)
    if var_thick_flag:
        dataloader_testing = DataLoader(dataset, batch_size,
                                         shuffle=False, num_workers=1, pin_memory=False,drop_last=False,collate_fn=dataset.mine_collate_fn)
    else:
        dataloader_testing = DataLoader(dataset, batch_size,
                                        shuffle=False, num_workers=1, pin_memory=False, drop_last=False)
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
    quickdraw_dir = '/media/data/QuickDraw-clean'
    var_thick_flag = True
    np.random.seed(100)
    random.seed(100)

    dataloader_training = getQuickDrawTrain(quickdraw_dir,batch_size=16,var_thick_flag=var_thick_flag)
    import time
    import torch
    device = torch.device('cuda')
    start_time = time.time()
    for i,sample_batch in enumerate(dataloader_training):
        stroke_batch = sample_batch#if var_thick_flag: get list[d*d] for [1,128,128] else: get [b,1,128,128] Tensor
        print('the dim of stroke_batch',len(stroke_batch))
        if var_thick_flag:
            del_white_num = 0
            for sub_idx in range(len(stroke_batch)):
                sub_stroke_batch = stroke_batch[sub_idx-del_white_num]
                all_white_flag = np.all(sub_stroke_batch == 255)#判断是否全部元素是255
                if all_white_flag:
                    del stroke_batch[sub_idx-del_white_num]
                    del_white_num += 1
            stroke_batch = torch.tensor(np.array(stroke_batch),dtype=torch.float)

        print('the dim of stroke_batch',stroke_batch.shape)
        batch_dim = stroke_batch.shape[0]

        show_array = np.array(stroke_batch[0, 0, :, :])
        plt.imshow(show_array, cmap='gray')
        # plt.savefig('./stroke_{}.png'.format(i))
        plt.show()

        if i > 10:
            break
        # stroke_tensor = stroke_batch.to(device)
        # imageio.imwrite('./stroke_{}.png'.format(i), sample_batch[0,0,:,:])
        # for m in range(2):
        #     show_array = np.array(stroke_batch[m*8, 0, :, :])
        #     plt.imshow(show_array, cmap='gray')
        #     # plt.savefig('./stroke_{}.png'.format(i))
        #     plt.show()
        # break
        # if i % 50 == 0:
        #     show_array = np.array(stroke_batch[0, 0, :, :])
        #     plt.imshow(show_array, cmap='gray')
        #     # plt.savefig('./stroke_{}.png'.format(i))
        #     plt.show()
        # print(time.time() - start_time)
        # start_time = time.time()
        # if i > 100:
        #     break
    # load_dataset_multi_object(quickdraw_dir)
