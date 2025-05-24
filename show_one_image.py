import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # x = np.arange(-1,1,0.01)
    # print(x)
    # f = 1/(1+)
    # image_path = 'sample_inputs/clean_line_drawings/elephant.png'
    image_path = '1366.png'
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    plt.imshow(img)
    plt.show()

    img_size = cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA)
    img_array = np.array(img_size)
    img_array = np.where(img_array < 200,0,255)
    plt.imshow(img_array)
    plt.show()

    img_size_2 = cv2.resize(img_size,(512,512),interpolation=cv2.INTER_AREA)
    img_array_2 = np.array(img_size_2)
    img_array_2 = np.where(img_array_2 < 200,0,255)
    plt.imshow(img_array_2)
    plt.show()

    a = 1