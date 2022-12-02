import random
import time
from tkinter.simpledialog import askstring, askinteger, askfloat
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk



class Convex_Hull():
    def __init__(self):
        pass

    # 算面积
    def calc_area(self,a, b, c):
        """
        判断三角形面积
        """
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        return x1 * y2 + x3 * y1 + x2 * y3 - x3 * y2 - x2 * y1 - x1 * y3

    #  生成随机点
    def rand_point_set(self,n, range_min=0, range_max=101):
        try:
            return list(zip([random.uniform(range_min, range_max) for _ in range(n)],
                            [random.uniform(range_min, range_max) for _ in range(n)]))
        except IndexError as e:
            print("\033[31m" + ''.join(e.args) + "\n输入范围有误！" + '\033[0m')

    def AreaOfUp(self,left, right, lists, boundary):
        area_max = 0
        max_point = ()
        for item in lists:
            if item == left or item == right:
                continue
            else:
                max_point = item if Object.calc_area(left, right, item) > area_max else max_point
                area_max = Object.calc_area(left, right, item) if Object.calc_area(left, right, item) > area_max else area_max
        if area_max != 0:
            boundary.append(max_point)
            Object.AreaOfUp(left, max_point, lists, boundary)
            Object.AreaOfUp(max_point, right, lists, boundary)

    def AreaOfDown(self,left, right, lists, boundary):
        area_max = 0
        max_point = ()
        for item in lists:
            if item == left or item == right:
                continue
            else:
                max_point = item if Object.calc_area(left, right, item) < area_max else max_point
                area_max = Object.calc_area(left, right, item) if Object.calc_area(left, right, item) < area_max else area_max
        if area_max != 0:
            boundary.append(max_point)
            Object.AreaOfDown(left, max_point, lists, boundary)
            Object.AreaOfDown(max_point, right, lists, boundary)

    def order_border(self,lists):
        lists.sort()
        first_x, first_y = lists[0]  # 最左边的点
        last_x, last_y = lists[-1]  # 最右边的点
        list_border_up = []  # 上半边界
        for item in lists:
            x, y = item
            if y > max(first_y, last_y):
                list_border_up.append(item)
            if min(first_y, last_y) < y < max(first_y, last_y):
                if Object.calc_area(lists[0], lists[-1], item) > 0:
                    list_border_up.append(item)
                else:
                    continue
        list_border_down = [_ for _ in lists if _ not in list_border_up]  # 下半边界
        list_end = list_border_up + list_border_down[::-1]  # 最终顺时针输出的边界点
        return list_end

    """可视化"""

    def print_integer(self):
        res = askinteger("Spam", "Egg count", initialvalue=12 * 12)
        return res

    def display(self,list_points, boundary):
        root = Tk()
        root.config(bg='#87CEEB')
        root.title("202018526高树林的凸包可视化")
        cv = Canvas(root, bg="white", width=800, height=800)
        cv.pack()
        # x轴
        for i in range(11):
            x = 100 + (i * 60)
        # y轴
        for i in range(6):
            y = 500 - (i * 80)
        scaled = []
        all_point = []
        for x, y in boundary:
            scaled.append((100 + 6 * x, 500 - 8 * y / 5))
        for x, y in list_points:
            all_point.append((100 + 6 * x, 500 - 8 * y / 5))
        scaled.append((boundary[0][0] * 6 + 100, 500 - 8 * boundary[0][1] / 5))
        cv.create_line(scaled, fill='green')
        for x, y in all_point:
            cv.create_oval(x - 6, y - 6, x + 6, y + 6, width=1, outline='black', fill='red')
        cv.create_text(350, 560, text='\t高树林使用Tkinter做的可视化', fill='black', font='SimHei 20 bold')
        root.mainloop()

    def main(self):
        """
        :return: 所有点
        """
        root = tk.Tk()
        tk.Button(root, text='取一个整数', command=Object.print_integer).pack()
        # inputs = list(map(int, input().split()))
        inputs = Object.print_integer()
        return Object.rand_point_set(inputs)


if __name__ == "__main__":
    Object = Convex_Hull()
    list_points = Object.main()  # 所有点
    # print(list_points)
    list_points.sort()
    border_points = []  # 边界点集
    Object.AreaOfUp(list_points[0], list_points[-1], list_points, border_points)  # 上边界点集
    Object.AreaOfDown(list_points[0], list_points[-1], list_points, border_points)  # 下边界点集
    border_points.append(list_points[0])
    border_points.append(list_points[-1])  # 将首尾两个点添加到边界点集中
    print(Object.order_border(border_points))  # 顺时针输出边界点
    Object.display(list_points, Object.order_border(border_points))