# -*- coding: utf-8 -*-
# 单文件版本
from __future__ import unicode_literals
import random
import time

try:
    from tkinter import *
except ImportError:
    from Tkinter import *
import matplotlib.image as mpimg  # 用于显示图片
import numpy


class Block(object):
    def __init__(self, mmap, x, y, direction=None):
        super(Block, self).__init__()
        # 初始定义四个方向都没有堵塞
        self.walls = [True, True, True, True]  # top,right,bottom,left
        self.mmap = mmap
        if mmap:
            mmap.mmap[x][y] = self
        self.x, self.y = x, y
        # 根据算法设置堵塞的墙
        if direction is not None:
            direction = (direction + 2) % 4
            self.walls[direction] = False

    def __unicode__(self):
        return "%s" % [1 if e else 0 for e in self.walls]

    def __str__(self):
        return unicode(self).encode('utf-8')

    # 根据已有点的位置坐标，获取周围四个方向的坐标
    def get_next_block_pos(self, direction):
        x = self.x
        y = self.y
        if direction == 0:  # Top
            y -= 1
        elif direction == 1:  # Right
            x += 1
        if direction == 2:  # Bottom
            y += 1
        if direction == 3:  # Left
            x -= 1
        return x, y

    def get_next_block(self):
        directions = list(range(4))  # 四个方向
        random.shuffle(directions)  # 随机找一个方向
        for direction in directions:  # 每个方向遍历查找
            x, y = self.get_next_block_pos(direction)  # 下一个方向的坐标
            if x >= self.mmap.max_x or x < 0 or y >= self.mmap.max_y or y < 0:
                continue  # 可以往下走
            if self.mmap.mmap[x][y]:  # if walked
                continue  # 走了
            self.walls[direction] = False  # 此路不通
            return Block(self.mmap, x, y, direction)  # 记住堵塞的位置
        return None


class Map(object):
    def __init__(self):
        super(Map, self).__init__()

    # 重画地图
    def reset_map(self):
        self.gen_map(self.max_x, self.max_y)

    def reset_map_changesize(self, chang, kuan):
        self.max_x = chang
        self.max_y = kuan
        self.gen_map(self.max_x, self.max_y)

    # 生成地图
    def gen_map(self, max_x=10, max_y=10):
        self.max_x, self.max_y = max_x, max_y
        self.mmap = [[None for j in range(self.max_y)] for i in range(self.max_x)]
        self.solution = []  # 定义路线的列表
        block_stack = [Block(self, 0, 0)]  # a unused block
        while block_stack:
            block = block_stack.pop()
            next_block = block.get_next_block()
            if next_block:
                block_stack.append(block)
                block_stack.append(next_block)
                # 如果到了终点，开始循环加载路线图
                if next_block.x == self.max_x - 1 and next_block.y == self.max_y - 1:  # is end
                    for o in block_stack:
                        self.solution.append((o.x, o.y))  # 将正确的路线记录下来

    # python2用的
    def __unicode__(self):
        out = ""
        for y in range(self.max_y):
            for x in range(self.max_x):
                out += "%s" % self.mmap[x][y]
            out += "\n"
        return out

    # python3用的
    def __str__(self):
        return unicode(self).encode('utf-8')


class DrawMap(object):
    def __init__(self, mmap, cell_width=10):
        super(DrawMap, self).__init__()
        self.mmap = mmap
        self.cell_width = cell_width

    # 地图尺寸
    def get_map_size(self):
        # width, height
        return (self.mmap.max_x + 2) * self.cell_width, (self.mmap.max_y + 2) * self.cell_width

    # 画地图的线
    def create_line(self, x1, y1, x2, y2, **kwarg):
        raise NotImplemented()

    # 行进路线
    def create_solution_line(self, x1, y1, x2, y2):
        self.create_line(x1, y1, x2, y2)

    def draw_start(self):
        raise NotImplemented()

    def draw_end(self):
        raise NotImplemented()

    def get_cell_center(self, x, y):
        w = self.cell_width
        return (x + 1) * w + w // 2, (y + 1) * w + w // 2

    def draw_solution(self):
        pre = (0, 0)  # 起点位置
        for o in self.mmap.solution:
            p1 = self.get_cell_center(*pre)  # 起始点位置
            p2 = self.get_cell_center(*o)  # 下一个点的位置
            self.create_solution_line(p1[0], p1[1], p2[0], p2[1])
            pre = o  # 起始点为上一个点

    def draw_cell(self, block):
        width = self.cell_width
        x = block.x + 1
        y = block.y + 1
        walls = block.walls
        if walls[0]:
            self.create_line(x * width, y * width, (x + 1) * width + 1, y * width)
        if walls[1]:
            self.create_line((x + 1) * width, y * width, (x + 1) * width, (y + 1) * width + 1)
        if walls[2]:
            self.create_line(x * width, (y + 1) * width, (x + 1) * width + 1, (y + 1) * width)
        if walls[3]:
            self.create_line(x * width, y * width, x * width, (y + 1) * width + 1)

    # 画图
    def draw_map(self):
        for y in range(self.mmap.max_y):
            for x in range(self.mmap.max_x):
                self.draw_cell(self.mmap.mmap[x][y])
        self.draw_start()
        self.draw_end()


class TKDrawMap(DrawMap):
    def __init__(self, mmap):
        super(TKDrawMap, self).__init__(mmap, cell_width=10)
        master = Tk()  # 调用tkinter
        master.title('202018526高树林的算法结课实验之迷宫')
        master.geometry('900x600')  # 这里的乘号不是 * ，而是小写英文字母 x
        btn_start = Button(master, text='start', command=self.draw_solution)
        btn_start.place(x=100, y=550)
        btn_start = Button(master, text='move', command=self.draw_solution)
        btn_start.place(x=200, y=550)#这里要改成可以沿着某条线走
        btn_change = Button(master, text='change', command=self.reset_map)
        btn_change.place(x=300, y=550)

        self.var = StringVar(master)
        self.var.set("choose width")  # initial value
        option = OptionMenu(master, self.var, "50", "40", "30")
        option.place(x=400, y=550)
        self.var2 = StringVar(master)
        self.var2.set("choose height")  # initial value
        option = OptionMenu(master, self.var2, "50", "40", "30")
        option.place(x=600, y=550)
        btn_size = Button(master, text="ok", command=self.ok)
        btn_size.place(x=800, y=550)

        width, height = self.get_map_size()  # 获取尺寸大小
        self.w = Canvas(master, width=width, height=width)  # 调用Canvas
        self.w.pack()  # 显示
        self.draw_map()
        mainloop()  # 循环

    def ok(self):
        change_size = self.var.get()
        change_size2 = self.var2.get()
        change_size = int(change_size)
        change_size2 = int(change_size2)

        self.mmap.reset_map_changesize(change_size, change_size2)
        self.w.delete('all')
        self.draw_map()

    # 按开始处红点按钮，开始画路线图
    def draw_start(self):
        r = self.cell_width // 3
        x, y = self.get_cell_center(0, 0)
        # start = self.w.create_oval(x - r, y - r, x + r, y + r, fill="red")
        start = self.w.create_rectangle(x - r, y - r, x + r, y + r, dash=(4, 4), fill='pink')
        self.w.tag_bind(start, '<ButtonPress-1>', lambda e: self.draw_solution())

    # 按结尾处红点按钮，更换地图
    def draw_end(self):
        r = self.cell_width // 3
        x, y = self.get_cell_center(self.mmap.max_x - 1, self.mmap.max_y - 1)
        end = self.w.create_oval(x - r, y - r, x + r, y + r, fill="red")
        self.w.tag_bind(end, '<ButtonPress-1>', lambda e: self.reset_map())

    # 重画地图
    def reset_map(self):
        self.mmap.reset_map()
        self.w.delete('all')
        self.draw_map()

    def create_line(self, x1, y1, x2, y2, **kwargs):
        self.w.create_line(x1, y1, x2, y2, **kwargs)

    def create_solution_line(self, x1, y1, x2, y2):
        self.create_line(x1, y1, x2, y2, fill="green")


def main():
    m = Map()  # 创建对象
    m.gen_map(50, 50)  # 调用方法，生成地图
    TKDrawMap(m)  # 画出路线图


if __name__ == '__main__':
    main()