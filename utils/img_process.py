import cv2
import matplotlib.pyplot as plt
import numpy as np

def img_pad_size(img):
    width,height = img.shape#图片的宽和高
    width_pad = 128 - width%128#宽度与128倍数之间的差值。
    height_pad = 128 - width%128
    padding = [[0,width_pad],
               [0,height_pad]]
    img = np.pad(img,padding,mode='constant',constant_values=255 if np.max(img)==255 else 1)
    # print(width_pad_x,width_pad_y)
    return img
def img_pad_size_set_width(img,sub_img_width):
    if len(img.shape) == 2:
        img_width,img_height = img.shape
    elif len(img.shape) == 3:
        img_width,img_height,_ = img.shape
    else:
        raise ValueError('the shape is error')
    width_divi_num = img_width//sub_img_width+1
    height_divi_num = img_height//sub_img_width+1
    width_pad = width_divi_num*sub_img_width-img_width#宽度上需要补充的像素宽度
    height_pad = height_divi_num*sub_img_width-img_width#高度上需要补充的像素宽度
    padding = [[width_pad//2,width_pad-width_pad//2],
               [height_pad//2,height_pad-height_pad//2]]
    img = np.pad(img,padding,mode='constant',constant_values=255 if np.max(img)==255 else 0)
    return img
def get_big_pad_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img_clip = np.where(img<200,0,255)#[872,872]
    img_clip = img_pad_size(img_clip)
    img_clip_ = img_clip[...,None]#[872,872] -> [872,872,1]
    img_clip_ = np.concatenate((img_clip_,img_clip_,img_clip_),axis=-1)#[872,872,1] -> [872,872,3]
    return img_clip_
def get_big_pad_img_set_width(img_path,sub_img_width=100):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:,:,0]
    img = np.where(img<200,0,255)#[872,872]
    print('original image shape : ',img.shape)
    img_clip = img_pad_size_set_width(img,sub_img_width)
    print('after add padding,the image shape : ',img_clip.shape)
    if len(img_clip.shape) == 2:
        img_clip_ = img_clip[...,None]#[872,872] -> [872,872,1]
        img_clip_ = np.concatenate((img_clip_,img_clip_,img_clip_),axis=-1)#[872,872,1] -> [872,872,3]
        return img_clip_
    else:
        return img_clip
if __name__ == '__main__':
    img_path = '/data/lgf/rl_draw_image/rl_draw_edge_render_change_error_new_state/sample_inputs/clean_line_drawings/img_1588.png'
    img = get_big_pad_img_set_width(img_path,sub_img_width=110)
