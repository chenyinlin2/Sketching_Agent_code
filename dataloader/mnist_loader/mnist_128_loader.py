import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import os

class Mnist128(Dataset):
    def __init__(self,dataset_base_dir,datakind='train',img_size=128):#datakind:[train,test,val]
        self.img_size = img_size#绘制的图片的大小
        self.data_kind = datakind
        self.data_base_dir = os.path.join(dataset_base_dir,self.data_kind)
        self.img_name_list = load_img_name_list(dataset_base_dir,datakind)

    def __getitem__(self, idx):
        img_name = self.img_name_list[idx]
        img_file_path = os.path.join(self.data_base_dir, img_name)#三通道彩色图像文件file路径
        img_data = Image.open(img_file_path)
        img_data_array = np.array(img_data)
        if img_data_array.shape[0] != self.img_size:
            img_data_array = cv2.resize(img_data_array, (self.img_size, self.img_size),interpolation=cv2.INTER_AREA)
        img_data_array = 255 - np.array(img_data_array)
        if len(img_data_array.shape) == 2:
            img_data_array = np.repeat(img_data_array[np.newaxis,...],3,axis=0)
        return img_data_array

    def __len__(self):
        return len(self.img_name_list)

def load_img_name_list(dataset_base_dir, datakind):
    img_name_path = os.path.join(dataset_base_dir,datakind,'img_name.txt')
    img_name_list = np.loadtxt(img_name_path, dtype=str)
    return img_name_list
def getMnistTrain(data_path='/data/DATA/MNIST/process_raw_128',batch_size=48,img_size=128):
    data_kind = 'train'
    dataset = Mnist128(data_path,data_kind,img_size=img_size)
    dataloader_training = DataLoader(dataset, batch_size,
                                     shuffle=True, num_workers=16, pin_memory=True,drop_last=True)
    return dataloader_training


def getMnistTest(dataset_dir='/data/DATA/MNIST/process_raw_128',batch_size=1,img_size=128,shuffle=False):#路径 | batch | 图像大小 | 线段宽度
    data_kind = 'test'
    dataset = Mnist128(dataset_dir,data_kind,img_size=img_size)
    dataloader_testing = DataLoader(dataset, batch_size,
                                     shuffle=shuffle, num_workers=1, pin_memory=False,drop_last=False)
    return dataloader_testing

if __name__ == '__main__':
    data_path = '/data/DATA/MNIST/process_raw_128'
    train_dataloader = getMnistTest(data_path)
    for i,img in enumerate(train_dataloader):
        img_show = img[0,:,:,:].permute(1,2,0).detach().cpu()
        img_show_array = img_show.numpy()
        plt.subplot(1,2,1)
        plt.imshow(img_show_array)
        plt.show()
        if i > 5:
            exit()

