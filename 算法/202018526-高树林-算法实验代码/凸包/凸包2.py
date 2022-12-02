import math
import matplotlib.pyplot as plt
import numpy as np


class Convex_Hull():
    def __init__(self):
        pass

    def atan(self, point, y, x):
        x = x - point[0]
        y = y - point[1]
        if x == 0 and y == 0:
            return 0
        point = (5, 0)  # 表示x轴的向量，随便取
        cos = (point[0] * x + point[1] * y) / (math.sqrt(point[0] ** 2 + point[1] ** 2) * math.sqrt(x ** 2 + y ** 2))
        return np.arccos(cos) * (180 / math.pi)

    def angle_sort(self, p0, points):
        dic = {}
        for point in points:
            angle = self.atan(p0, point[1], point[0])
            dic[point] = angle
        points = [k[0] for k in sorted(dic.items(), key=lambda x: x[1])]  # ,reverse=True
        return points

    def cross_product(self, a, b, c):
        '''判断点c在由点a,b构成的向量的那一侧'''
        result = a[0] * b[1] - a[1] * b[0] + b[0] * c[1] - b[1] * c[0] + c[0] * a[1] - c[1] * a[0]
        if result < 0:
            return False  # 点c在向量ab右边 返回False
        else:
            return True  # 点c在向量ab左边  返回True

    def draw(self, x, y, x0, y0):
        plt.figure(figsize=(10, 10))
        plt.scatter(x, y)
        plt.plot(x0, y0)
        plt.xlim(-1, 11)
        plt.ylim(-1, 6)
        i = 0
        plt.show()


if __name__ == '__main__':
    Object = Convex_Hull()
    key = 3
    num = 300
    level = 10
    vertical = 5
    seed = np.random.RandomState(key)
    seed2 = np.random.RandomState(key + 1)
    Z1 = seed.rand(num, 1) * level  # 生成点集
    Z2 = seed2.rand(num, 1) * vertical
    Z = np.concatenate([Z1, Z2], axis=1)
    lists_points = [tuple(i) for i in Z]

    # 起点为y坐标最小的点
    ymin = min(lists_points, key=lambda x: x[1])[1]
    start = min([i for i in lists_points if i[1] == ymin], key=lambda x: x[0])
    boundary = []
    lists_points = Object.angle_sort(start, lists_points)
    boundary.append(lists_points[0])
    boundary.append(lists_points[1])
    i = 2
    while len(boundary) != 0 and i != len(lists_points):
        if Object.cross_product(boundary[len(boundary) - 2], boundary[len(boundary) - 1], lists_points[i]):
            boundary.append(lists_points[i])
            i += 1
        else:
            boundary.pop()
            if len(boundary) < 2:
                boundary.append(lists_points[i])
                i += 1
            continue
    boundary.append(boundary[0])
    x = [i[0] for i in lists_points]
    y = [i[1] for i in lists_points]
    x0 = [i[0] for i in boundary]
    y0 = [i[1] for i in boundary]
    Object.draw(x, y, x0, y0)