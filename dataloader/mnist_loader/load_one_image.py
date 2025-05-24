import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
if __name__ == '__main__':
    image_path = '/data/DATA/MNIST/process_raw/test/9/9.png'
    image = cv2.imread(image_path)
    plt.imshow(image)
    plt.show()
    #------------------------------------
    image_2 = np.where(image<128,0,255)
    plt.imshow(image_2)
    plt.show()
    #------------------------------------
    # image_2_128 = cv2.resize(image,(128,128),interpolation=cv2.INTER_LINEAR)
    # plt.imshow(image_2_128)
    # plt.show()
    #------------------------------------
    image_128 = cv2.resize(image,(128,128),interpolation=cv2.INTER_LINEAR)
    plt.imshow(image_128)
    plt.show()
    #------------------------------------
    image_128_2 = np.where(image_128<128,0,255)
    plt.imshow(image_128_2)
    plt.show()
    #------------------------------------
    a = 1