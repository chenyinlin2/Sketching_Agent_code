import struct
import numpy as np
import os
import cv2
#解析图像数据
def decode_idx3_ubyte(idx3_ubyte_file):
    with open(idx3_ubyte_file, 'rb') as f:
        print('解析文件：', idx3_ubyte_file)
        fb_data = f.read()

    offset = 0
    fmt_header = '>iiii'    # 以大端法读取4个 unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, fb_data, offset)
    print('魔数：{}，图片数：{}'.format(magic_number, num_images))
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(num_rows * num_cols) + 'B'

    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        im = struct.unpack_from(fmt_image, fb_data, offset)
        images[i] = np.array(im).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images
#解析标签数据
def decode_idx1_ubyte(idx1_ubyte_file):
    with open(idx1_ubyte_file, 'rb') as f:
        print('解析文件：', idx1_ubyte_file)
        fb_data = f.read()

    offset = 0
    fmt_header = '>ii'  # 以大端法读取两个 unsinged int32
    magic_number, label_num = struct.unpack_from(fmt_header, fb_data, offset)
    print('魔数：{}，标签数：{}'.format(magic_number, label_num))
    offset += struct.calcsize(fmt_header)
    labels = []

    fmt_label = '>B'    # 每次读取一个 byte
    for i in range(label_num):
        labels.append(struct.unpack_from(fmt_label, fb_data, offset)[0])
        offset += struct.calcsize(fmt_label)
    return labels

def check_folder(folder):
    """检查文件文件夹是否存在，不存在则创建"""
    if not os.path.exists(folder):
        os.makedirs(folder,exist_ok=True)
        print(folder)
    else:
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)


def export_img(exp_dir, img_ubyte, lable_ubyte,img_name_save):
    """
    生成数据集
    """
    check_folder(exp_dir)
    images = decode_idx3_ubyte(img_ubyte)
    labels = decode_idx1_ubyte(lable_ubyte)

    save_var = '\n'
    f = open(img_name_save, "w")
    img_file_path = []
    nums = len(labels)
    for i in range(nums):
        img_dir = os.path.join(exp_dir, str(labels[i]))
        check_folder(img_dir)
        img_file = os.path.join(img_dir, str(i)+'.png')
        imarr = images[i]

        imarr = cv2.resize(imarr,(128,128),interpolation=cv2.INTER_LINEAR)
        imarr = np.where(imarr<128,0,255)

        cv2.imwrite(img_file, imarr)
        img_file_path.append(str(labels[i])+'/'+str(i)+'.png')
    f.write(save_var.join(img_file_path))
    f.close()

def parser_mnist_data(data_dir):
    train_dir = os.path.join(data_dir, 'process_raw_128/train')
    train_img_ubyte = os.path.join(data_dir, 'raw/train-images-idx3-ubyte')
    train_label_ubyte = os.path.join(data_dir, 'raw/train-labels-idx1-ubyte')
    train_img_name_save_path = os.path.join(data_dir,'process_raw_128/train/img_name.txt')
    print(train_dir)
    print(train_img_ubyte)
    print(train_label_ubyte)
    export_img(train_dir, train_img_ubyte, train_label_ubyte,train_img_name_save_path)

    test_dir = os.path.join(data_dir, 'process_raw_128/test')
    test_img_ubyte = os.path.join(data_dir, 'raw/t10k-images-idx3-ubyte')
    test_label_ubyte = os.path.join(data_dir, 'raw/t10k-labels-idx1-ubyte')
    test_img_name_save_path = os.path.join(data_dir,'process_raw_128/test/img_name.txt')
    print(test_dir)
    print(test_img_ubyte)
    print(test_label_ubyte)
    export_img(test_dir, test_img_ubyte, test_label_ubyte,test_img_name_save_path)


if __name__ == '__main__':
    data_dir = '/data/DATA/MNIST'
    parser_mnist_data(data_dir)