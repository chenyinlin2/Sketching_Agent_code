import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
#使用数值积分的方法来计算
# 定义贝塞尔曲线函数
def bezier(t, P0, P1, P2):
    B = (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2
    return B
# 定义曲线导数函数
def derivative(t, P0, P1, P2):
    dX = 2 * (1 - t) * (P1 - P0) + 2 * t * (P2 - P1)
    return np.linalg.norm(dX)
def get_bezier_length(p0,p1,p2):
    # 使用数值积分方法计算曲线的长度
    curve_length, _ = quad(derivative, 0, 1, args=(p0,p1,p2))
    return curve_length
#对曲线进行离散化的方式
def get_bezier_length_discrete(p0,p1,p2,n=100):
    # 初始化总长度
    curve_length = 0
    t = np.linspace(0, 1, n)
    # 计算每个小线段的长度并累加
    for i in range(1, n):
        t0, t1 = t[i - 1], t[i]
        dt = t1 - t0
        p0_t = (1 - t0) ** 2 * p0 + 2 * (1 - t0) * t0 * p1 + t0 ** 2 * p2
        p1_t = (1 - t1) ** 2 * p0 + 2 * (1 - t1) * t1 * p1 + t1 ** 2 * p2
        segment_length = np.linalg.norm(p1_t - p0_t)
        curve_length += segment_length
    return curve_length
if __name__ == '__main__':
    from Renderer.stroke_gen import draw_circle_2
    import time
    # 二次贝塞尔曲线的参数
    # p0 = np.random.rand(2) # 起点
    # p1 = np.random.rand(2)  # 控制点
    # p2 = np.random.rand(2) # 终点
    p0 = np.array([0,0])
    p1 = np.array([0,0])
    p2 = np.array([1.0,1.0])
    curve_img = draw_circle_2(np.concatenate([p0,p1,p2],axis=0))
    plt.imshow(curve_img)
    plt.show()
    start_time = time.time()
    print("二次贝塞尔曲线的长度：", get_bezier_length(p0,p1,p2))
    end_time_1 = time.time()
    print(end_time_1 - start_time)
    print("二次贝塞尔曲线的长度：", get_bezier_length_discrete(p0,p1,p2))
    end_time_2 = time.time()
    print(end_time_2 - end_time_1)

