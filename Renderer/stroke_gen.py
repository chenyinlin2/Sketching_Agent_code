import cv2
import numpy as np
from PIL import Image
def normal(x, width):
    return (int)(x * (width - 1) + 0.5)
def draw_circle_2_gif(f, width=128,stroke_pixel_length=5):
    '''如果在return处增加resize会使得1和0之间被平均，其中的深浅度也会被平均，感觉有一定影响。所以输出多大的画布，中间就不进行扩大'''
    '''生成更短小的笔划，返回短小笔划的矩阵和笔划端点，最终连接起来生成'''
    '''divide_num : 当绘制的点超过15个，则将其视作一个小标点'''
    if f.shape[0] == 8:
        x0, y0, x1, y1, x2, y2,z0,z2 = f
    elif f.shape[0] == 6:
        x0, y0, x1, y1, x2, y2 = f
        z0, z2 = 0,0
    elif f.shape[0] == 7:
        x0, y0, x1, y1, x2, y2,z0 = f
        z0 = z0*5/128
        z2 = z0
    else:
        raise ValueError("f的维度不正确")
    w0,w2 = 1,1
    # z0, z2, w0, w2 = 0,0,1,1#z0,z1表示线的宽度，w0，w2表示线的颜色
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1

    x0 = normal(x0, width)
    x1 = normal(x1, width)
    x2 = normal(x2, width)

    y0 = normal(y0, width)
    y1 = normal(y1, width)
    y2 = normal(y2, width)

    z0 = (int)(1 + z0 * width)
    z2 = (int)(1 + z2 * width)

    canvas = np.zeros([width , width]).astype('float32')
    tmp = 1. / 100
    small_strokes = {'stroke_canvas':[],'end_point':[]}
    last_x,last_y = x0,y0
    new_canvas = np.zeros([width, width]).astype('float32')
    for i in range(100):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)#(y,x)是中心坐标，z是圆的半径，w是圆边界线的颜色
        cv2.circle(new_canvas, (y, x), z, w, -1)#(y,x)是中心坐标，z是圆的半径，w是圆边界线的颜色

        draw_pixel_dis = get_l2_dis(np.array([y,x]),np.array([last_y,last_x]),width)
        if draw_pixel_dis > stroke_pixel_length:
            last_x, last_y = x, y
            small_strokes['stroke_canvas'].append(new_canvas)
            small_strokes['end_point'].append(np.array([y,x]))
            new_canvas = np.zeros([width, width]).astype('float32')
        elif i == 99:
            last_x, last_y = x, y
            small_strokes['stroke_canvas'].append(new_canvas)
            small_strokes['end_point'].append(np.array([y,x]))
            new_canvas = np.zeros([width, width]).astype('float32')
    return canvas,small_strokes


def get_l2_dis(point_1,point_2,width):#计算两个点之间的像素距离
    pixel_dis = np.sqrt(np.sum((point_2-point_1)**2))
    return pixel_dis
def draw_circle(f, width=128):
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    w0,w2 = 1,1
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    # print(x1,y1)
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)

    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)

    z0 = (int)(1 + z0 * width // 2)
    z2 = (int)(1 + z2 * width // 2)

    canvas = np.zeros([width * 2, width * 2]).astype('float32')
    tmp = 1. / 100
    for i in range(100):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        # cv2.circle(canvas, (y, x), 1, w, -1)#(y,x)是中心坐标，z是圆的半径，w是圆边界线的颜色
        cv2.circle(canvas, (y, x), z, w, -1)#(y,x)是中心坐标，z是圆的半径，w是圆边界线的颜色
    return 1 - cv2.resize(canvas, dsize=(width, width))
def draw_circle_2(f, width=128):
    '''如果在return处增加resize会使得1和0之间被平均，其中的深浅度也会被平均，感觉有一定影响。所以输出多大的画布，中间就不进行扩大'''
    if f.shape[0] == 8:
        x0, y0, x1, y1, x2, y2,z0,z2 = f
        z0 = z0*5/128
        z2 = z2*5/128
    elif f.shape[0] == 6:
        x0, y0, x1, y1, x2, y2 = f
        z0, z2 = 0,0
    elif f.shape[0] == 7:
        x0, y0, x1, y1, x2, y2,z0 = f
        z0 = z0*5/128
        z2 = z0
    else:
        raise ValueError("f的维度不正确")
    w0,w2 = 1,1
    # z0, z2, w0, w2 = 0,0,1,1#z0,z1表示线的宽度，w0，w2表示线的颜色
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1

    x0 = normal(x0, width)
    x1 = normal(x1, width)
    x2 = normal(x2, width)

    y0 = normal(y0, width)
    y1 = normal(y1, width)
    y2 = normal(y2, width)

    z0 = (int)(1 + z0 * width)
    z2 = (int)(1 + z2 * width)

    canvas = np.zeros([width , width]).astype('float32')
    tmp = 1. / 100
    for i in range(100):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)#(y,x)是中心坐标，z是圆的半径，w是圆边界线的颜色
    return 1 - canvas

def draw_circle_for256(f, width=128):
    '''如果在return处增加resize会使得1和0之间被平均，其中的深浅度也会被平均，感觉有一定影响。所以输出多大的画布，中间就不进行扩大'''
    x0, y0, x1, y1, x2, y2 = f
    z0, z2 = 0,0
    w0,w2 = 1,1
    # z0, z2, w0, w2 = 0,0,1,1#z0,z1表示线的宽度，w0，w2表示线的颜色
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1

    x0 = normal(x0, width)
    x1 = normal(x1, width)
    x2 = normal(x2, width)

    y0 = normal(y0, width)
    y1 = normal(y1, width)
    y2 = normal(y2, width)

    z0 = (int)(1 + z0 * width)
    z2 = (int)(1 + z2 * width)

    canvas = np.zeros([width , width]).astype('float32')
    tmp = 1. / 200
    for i in range(200):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)#(y,x)是中心坐标，z是圆的半径，w是圆边界线的颜色
    return 1 - canvas

def draw_line(f, width=128):
    '''如果在return处增加resize会使得1和0之间被平均，其中的深浅度也会被平均，感觉有一定影响。所以输出多大的画布，中间就不进行扩大'''
    if f.shape[0] == 8:
        x0, y0, x1, y1, x2, y2,z0,z2 = f
    elif f.shape[0] == 6:
        x0, y0, x1, y1, x2, y2 = f
        z0, z2 = 0,0
    elif f.shape[0] == 7:
        x0, y0, x1, y1, x2, y2,z0 = f
        z2 = z0
    else:
        raise ValueError("f的维度不正确")
    z0,z2 = 0,0
    w0,w2 = 1,1
    # z0, z2, w0, w2 = 0,0,1,1#z0,z1表示线的宽度，w0，w2表示线的颜色
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1

    x0 = normal(x0, width)
    x1 = normal(x1, width)
    x2 = normal(x2, width)

    y0 = normal(y0, width)
    y1 = normal(y1, width)
    y2 = normal(y2, width)

    z0 = (int)(1 + z0 * width)
    z2 = (int)(1 + z2 * width)

    canvas = np.zeros([width , width]).astype('float32')
    tmp = 1. / 100
    for i in range(100):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)

        t = (i+1) * tmp
        x_next = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y_next = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        cv2.line(canvas, (y, x),(y_next,x_next), w,z)#(y,x)是中心坐标，z是画线的宽度，w是线的颜色
        # print(y,x,'|',y_next,x_next)
        # cv2.circle(canvas, (y, x), z, w, -1)#(y,x)是中心坐标，z是圆的半径，w是圆边界线的颜色
    return 1 - canvas

if __name__ == '__main__':
    f = np.random.uniform(0, 1, 10)
    print(f)
    img = draw_circle(f)
    img2 = draw_line(f)
    # img_PIL = Image.fromarray(img)
    # img2_PIL = Image.fromarray(img2)
    # img_PIL.show()
    # img2_PIL.show()
    pass
    cv2.imshow('img',img)
    cv2.imshow('img2',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#作者您好。感谢您开源的代码，使我对于文章的理解更加深刻。但是我还是有一个不太理解的点想请教一下，为什么要使
# 用渲染网络呢？我看代码中渲染网络的输入和输出完全是可以自己定义好的。为什么不直接知道动作后，根据这个draw()函数得到应该在原画布上增加的笔划，反而要再训练出来一个神经网络呢？