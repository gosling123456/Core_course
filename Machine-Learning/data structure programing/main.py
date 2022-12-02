# coding=utf-8
import sys
import time as paused
import os

"""读取txt文件，并提取相应的单词"""


def text_processsing():
    text = (open('English.txt', "r", encoding='utf-8').read()).lower()  # 首先将所有单词转化为小写，方便后期统计同一单词的频次
    result_test = ''
    for i in text:  # 将标点符号（除了缩写时用到的'）转化为空字符，以便后期删除。
        if i in [',', ';', ':', '.', '!', '?', "'", '"', '-', '/', '(', ')', '[', ']', '*', '\n', '\t']:
            result_test += ' '
        else:
            result_test += i
    words = result_test.split(' ')  # 用split函数分开单词
    words = [x.strip() for x in words if x.strip() != '']  # 删除所有的空字符
    return words

def frequency(WordsList):
    ResultFrequency = {}
    SetOfWordsList = list(set(WordsList))
    for i in range(0, len(set(WordsList))):
        ResultFrequency.setdefault(SetOfWordsList[i], 0)
    for i in range(len(WordsList)):
        if WordsList[i] in ResultFrequency.keys():
            ResultFrequency[WordsList[i]] += 1
    return ResultFrequency


def SortFrequency(dic):
    dic_order = sorted(dic.items(), key=lambda x: x[1], reverse=True)  # 按字典集合中，每一个元组的第二个元素排列。
    return dict(dic_order)
    # from collections import Counter
    # a = Counter(text_prossing('English.txt'))


def addcode():
    f = open('user_name.txt', 'r', encoding='utf-8')
    sourceInLine = f.readlines()
    user_name = []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split('\t')
        user_name.append(temp2)
    user_name = [j for i in user_name for j in i]
    NewName = input('请输入你的姓名：')
    while NewName in user_name:
        print('用户名已存在，请重新输入！')
        os.system('cls')
        NewName = input('请输入你的姓名：')
    password1 = input('请设置密码（一般为学号）：')
    password2 = input('请确认密码：')
    while password2 != password1:
        print('两次密码不同，请核对后输入！')
        password1 = input('请设置密码（一般为学号）：')
        password2 = input('请确认密码：')
    with open("user_name.txt", encoding="utf-8", mode="a") as file:
        file.write('\n' + NewName)
    with open("user_password.txt", encoding="utf-8", mode="a") as file:
        file.write('\n' + password2)
    print('账号密码设置成功，按任意键返回登陆页面！')
    press = input()
    if press != '':
        login()
        os.system('cls')


def login():
    os.system('cls')
    print('                     比特单元英语词频统计系统\n\n\n')
    print('                      1：已有帐号，我要登陆！\n')
    print('                      2：我是小白，想要注册！\n')
    print('                      Any：我点错了，想要退出！\n')
    decision = input('请选择操作：')
    if decision == '1':
        os.system('cls')
        f = open('user_name.txt', 'r', encoding='utf-8')
        sourceInLine = f.readlines()
        user_name = []
        for line in sourceInLine:
            temp1 = line.strip('\n')
            temp2 = temp1.split('\t')
            user_name.append(temp2)
        user_name = [j for i in user_name for j in i]
        f = open('user_password.txt', 'r', encoding='utf-8')
        sourceInLine = f.readlines()
        user_password = []
        for line in sourceInLine:
            temp1 = line.strip('\n')
            temp2 = temp1.split('\t')
            user_password.append(temp2)
        user_password = [j for i in user_password for j in i]
        print('                     欢迎使用比特单元开发的英语词频统计系统(2020185班可登录）')
        name = input("请输入您的账号（姓名）：")
        if name in user_name:
            cnt = 3
            while cnt:
                password = input("请输入您的的密码（学号）：")
                if password not in user_password:
                    cnt -= 1
                    if cnt == 0:
                        print('您的输入有误，请您重新登录！')
                        login()
                    print('您的输入有误，请您重新输入！您还有%d次机会！' % cnt)
                elif user_password.index(password) == user_name.index(name):
                    print("                     登陆成功！正在为您准备菜单页面......")
                    paused.sleep(2)
                    menu()
                else:
                    cnt -= 1
                    if cnt == 0:
                        print('您的输入有误，请您重新登录！')
                        login()
                    print('您的输入有误，请您重新输入！您还有%d次机会！' % cnt)
        else:
            print('用户名错误，请您核对后输入！')
    if decision == '2':
        addcode()
    if decision == '3':
        print('感谢您的使用！')
        sys.exit()


def menu():
    os.system('cls')
    print("                     欢迎使用比特单元开发的英语词频统计系统菜单，在这里你可以做以下选择：\n\n\n")
    print("                      1.查找\n")
    print("                      2.排序\n")
    print("                      3.退出系统\n")
    choose = input('请输入操作对应的序号：')
    if choose == '1':
        FindMenu()
    elif choose == '2':
        SortMenu()
    elif choose == '3':
        print('感谢您的使用！')
        sys.exit()
    else:
        print('输入错误！正在为你返回主菜单......')
        paused.sleep(2)
        menu()
    selection = input('是否继续查询Y/N? ')
    if selection.lower() == 'yes' or 'y':
        menu()
    elif selection.lower() == 'no' or 'n':
        print('感谢您的使用！')
        sys.exit()
    else:
        print('输入错误，请重新输入！')


def LinkList():
    name = input('请输入你要查找的单词：')
    single_obj = SingleLinkList()
    ResultOfStatistics = SortFrequency(frequency(text_processsing()))
    for i in ResultOfStatistics.keys():
        single_obj.append(i, ResultOfStatistics[i])
    single_obj.traval(name)


def SequentialList():
    name = input('请输入你要查找的单词：')
    sequenceList = SequenceList()
    ResultOfStatistics = SortFrequency(frequency(text_processsing()))
    for key, value in ResultOfStatistics.items():
        sequenceList.append([key, value])
    sequenceList.traval(name)


def BST():
    name = input('请输入你要查找的单词：')
    time = []
    word = []
    ResultOfStatistics = SortFrequency(frequency(text_processsing()))
    for key, value in ResultOfStatistics.items():
        time.append(key)
        word.append(value)
    if name not in time:
        print('二叉排序树中没有该对象！请检查你的输入！正在为你返回主菜单......')
        paused.sleep(2)
        menu()
    middle = time[int(len(time) / 2)]
    root = BTNode(middle, None, None)
    if middle in time:
        time.pop(time.index(middle))
    bTree = BTree(root)
    for i in time:
        bTree.insert(i)
    bTree.printBTree(name)


def OpenArdess(num, word, hash_table, word_table):
    hash_index = num % len(hash_table)
    while True:
        if hash_table[hash_index] == "0":
            hash_table[hash_index] = num
            word_table[hash_index] = word
            break
        else:
            hash_index = (hash_index + 1) % len(hash_table)


def searchByOpenArdes():
    name = input('请输入你要查找的单词：')
    time = []
    word = []
    ResultOfStatistics = SortFrequency(frequency(text_processsing()))
    for key, value in ResultOfStatistics.items():
        word.append(key)
        time.append(value)
    hash_tables = ["0"] * len(ResultOfStatistics)
    word_table = ["0"] * len(ResultOfStatistics)
    for i in range(len(ResultOfStatistics)):
        OpenArdess(time[i], word[i], hash_tables, word_table)
    for i in range(len(word_table)):
        if name == word_table[i]:
            print('查找成功,%s出现的次数为%d,本次为第%d次查找' % (name, time[i], i + 1))
            return
    print('哈希表中没有该对象！请检查你的输入！正在为你返回主菜单......')
    paused.sleep(2)
    menu()


def BalanceTree():
    name = input('请输入你要查找的单词：')
    word_tree = Tree()
    word = []
    ResultOfStatistics = SortFrequency(frequency(text_processsing()))
    for key, value in ResultOfStatistics.items():
        word.append(key)
    for i in word:
        word_tree.head = word_tree.insert1(word_tree.head, i)
    for i in range(len(word_tree.print1())):
        if word_tree.print1()[i] == name:
            print('查找成功,%s出现的次数为%d,本次为第%d次查找' % (name, ResultOfStatistics[name], i + 1))
            return
    print('平衡二叉树中没有该对象！请检查你的输入！正在为你返回主菜单......')
    paused.sleep(2)
    menu()


def FindMenu():
    os.system('cls')
    print("                     欢迎使用比特单元开发的英语词频统计查找模块，在这里你可以做以下形式的查找：\n\n\n")
    print("                      1.基于链表的顺序查找\n")
    print("                      2.基于顺序表的折半查找\n")
    print("                      3.基于二叉排序树的查找\n")
    print("                      4.基于开放地址法的散列查找\n")
    print("                      5.基于平衡二叉树的查找\n")
    print("                      6.返回上一级\n")
    print("                      7.退出系统\n")
    choose = input('请输入操作对应的序号：')
    if choose == '1':
        LinkList()
    elif choose == '2':
        SequentialList()
    elif choose == '3':
        BST()
    elif choose == '4':
        searchByOpenArdes()
    elif choose == '5':
        BalanceTree()
    elif choose == '6':
        menu()
    elif choose == '7':
        print('感谢您的使用！')
        sys.exit()
    else:
        print('输入错误！请重新输入！')
        paused.sleep(2)
        FindMenu()


def Heap(time, word):
    for i in range(len(time)):
        # print('---')
        while i > 0 and time[(i - 1) // 2] < time[i]:
            # print(i)
            time[(i - 1) // 2], time[i] = time[i], time[(i - 1) // 2], i = (i - 1) // 2
            word[(i - 1) // 2], word[i] = word[i], word[(i - 1) // 2], i = (i - 1) // 2

    for i in range(len(time)):
        time[0], time[len(time) - i - 1] = time[len(time) - i - 1], time[0]  # 交换首尾位置
        word[0], word[len(time) - i - 1] = word[len(time) - i - 1], word[0]
        flag = 0
        while 1:
            x, y = flag * 2 + 1, flag * 2 + 2

            nums = {time[flag]: flag}
            if x < len(time) - i - 1:
                nums[time[x]] = x
            if y < len(time) - i - 1:
                nums[time[y]] = y
            temp = max(nums.keys())

            if time[flag] < temp:  # 当前节点与较大的那个交换位置,如果自己就是最大的,或者找到了尽头,则循环结束
                time[flag], time[nums[temp]] = time[nums[temp]], time[flag]
                word[flag], word[nums[temp]] = word[nums[temp]], word[flag]
                flag = nums[temp]
            else:
                break
    return time, word


def HeapSort():
    time = []
    word = []
    ResultOfStatistics = SortFrequency(frequency(text_processsing()))
    for key, value in ResultOfStatistics.items():
        word.append(key)
        time.append(value)
    time, word = Heap(time, word)
    print('按照大根堆排序的结果为：')
    for i in range(len(word) - 1, -1, -1):
        print('单词：', word[i], '频数：', time[i])


def Quicksort(time, word, i, j):
    if i >= j:
        return time, word
    pivot = time[i]
    pivotd = word[i]
    low = i
    high = j
    while i < j:
        while i < j and time[j] >= pivot:
            j -= 1
        time[i] = time[j]
        word[i] = word[j]
        while i < j and time[i] <= pivot:
            i += 1
        time[j] = time[i]
        word[j] = word[i]
    time[j] = pivot
    word[j] = pivotd
    Quicksort(time, word, low, i - 1)
    Quicksort(time, word, i + 1, high)
    return time, word


def questsort():
    time = []
    word = []
    ResultOfStatistics = SortFrequency(frequency(text_processsing()))
    for key, value in ResultOfStatistics.items():
        word.append(key)
        time.append(value)
    time, word = Quicksort(time, word, 0, len(word) - 1)
    print('按照词频大小快速排列的结果为：')
    for i in range(len(time) - 1, -1, -1):
        print('单词为：', word[i], '频数为：', time[i])


def ShellSort():
    time = []
    word = []
    ResultOfStatistics = SortFrequency(frequency(text_processsing()))
    for key, value in ResultOfStatistics.items():
        word.append(key)
        time.append(value)
    b = len(time)  # 列表长度
    gap = b // 2  # 初始步长设置为总长度的一半
    while gap >= 1:
        for i in range(b):
            j = i
            while j >= gap and time[j - gap] > time[j]:  # 在每一组里面进行直接插入排序
                time[j], time[j - gap] = time[j - gap], time[j]
                word[j], word[j - gap] = word[j - gap], word[j]
                j -= gap
        gap = gap // 2  # 更新步长
    print('按照词频大小希尔排列的结果为：')
    for i in range(len(time) - 1, -1, -1):
        print('单词为：', word[i], '频数为：', time[i])


def SortMenu():
    os.system('cls')
    print("                     欢迎使用比特单元开发的英语词频统计查找模块，在这里你可以做以下形式的排序：\n\n\n")
    print("                      1.对单词按照词频进行堆排序\n")
    print("                      2.对单词按照词频进行快速排序\n")
    print("                      3.对单词按照词频进行希尔排序n\n")
    print("                      4.返回上一级\n")
    print("                      5.退出系统\n")
    choose = input('请输入操作对应的序号：')
    if choose == '1':
        HeapSort()
    elif choose == '2':
        questsort()
    elif choose == '3':
        ShellSort()
    elif choose == '4':
        menu()
    elif choose == '5':
        print('感谢您的使用！')
        sys.exit()
    else:
        print('输入错误！请重新输入！')
        paused.sleep(2)
        SortMenu()


# 定义节点
class Node(object):
    def __init__(self, word, time):
        self.word = word
        self.time = time
        self.next = None


# 定义链表
class SingleLinkList(object):
    def __init__(self, node=None):
        self._head = node

    def is_empty(self):
        return self._head is None

    def traval(self, word):
        current = self._head  # 遍历结点
        i = 1
        try:
            while current.word != word and current.word is not None:
                current = current.next
                i += 1
            print('查找成功,%s出现的次数为%d,本次为第%d次查找' % (current.word, current.time, i))
            return
        except:
            print('链表中没有该对象！请检查你的输入！正在为你返回主菜单......')
            paused.sleep(2)
            menu()

    def append(self, word, time):
        node = Node(word, time)
        if self.is_empty():
            self._head = node
        else:
            current = self._head
            while current.next is not None:
                current = current.next
            current.next = node


# 定义顺序表
class SequenceList(object):
    ResultOfStatistics = SortFrequency(frequency(text_processsing()))

    def __init__(self, max=len(ResultOfStatistics)):
        self.max = max
        self.data = [None] * self.max
        self.num = 0

    def append(self, value):  # 在表尾部插入一个元素
        if self.num > self.max:
            print("list is full")
        else:
            self.data[self.num] = value
            self.num += 1

    def traval(self, value):  # 按值查找第一个等于该值得索引
        for j in range(self.num):
            if self.data[j][0] == value:
                print('第%d次查找，找到的单词和对应次数分别是：' % (j + 1), self.data[j][0], ':', self.data[j][1])
                break
            if j == self.num - 1:
                print('顺序表中没有该对象！请检查你的输入！正在为你返回主菜单......')
                paused.sleep(2)
                menu()


# 定义查找二叉树的节点
class BTNode:
    def __init__(self, data, left, right):
        self.data = data
        self.left = left
        self.right = right


class BTree:

    def __init__(self, root):
        self.root = root

    def insert(self, data):
        self.insertNode(data, self.root)

    def insertNode(self, data, btNode):
        if btNode is None:  # Reach the edge node, create a new node
            btNode = BTNode(data, None, None)

        elif data < btNode.data:  # search for the appropriate position
            if btNode.left is None:
                btNode.left = BTNode(data, None, None)

                return
            else:
                self.insertNode(data, btNode.left)

        elif data > btNode.data:  # search for the appropriate position
            if btNode.right is None:
                btNode.right = BTNode(data, None, None)

                return
            else:
                self.insertNode(data, btNode.right)

    def printBTreeImpl(self, btNode, data):
        ResultOfStatistics = SortFrequency(frequency(text_processsing()))
        if btNode is None:
            return
        elif btNode.data == data:
            print('查找成功,%s出现的次数为%d,' % (data, ResultOfStatistics[data]))
            return
        self.printBTreeImpl(btNode.left, data)
        self.printBTreeImpl(btNode.right, data)

    def printBTree(self, data):
        self.printBTreeImpl(self.root, data)
        return


class tree:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


class Tree:
    def __init__(self):
        self.head = None

    def print1(self):  # 层次输出二叉树的值
        a = []
        data = []
        if self.head is None:
            print("元素为空")
        else:
            a.append(self.head)
            while len(a) != 0:
                temp = a[0]
                data.append(temp.data)
                if temp.left is not None:
                    a.append(temp.left)
                if temp.right is not None:
                    a.append(temp.right)
                a.pop(0)
        return data

    def deep(self, head):  # 递归求某一个节点的深度
        if head is None:
            return 0
        else:
            return self.max1(self.deep(head.left), self.deep(head.right)) + 1

    def max1(self, a, b):
        if a < b:
            return b
        else:
            return a

    def LL(self, a):
        temp = a.left.right
        temp1 = a.left
        temp1.right = a
        a.left = temp
        return temp1

    def RR(self, a):
        temp = a.right.left
        temp1 = a.right
        temp1.left = a
        a.right = temp
        return temp1

    def LR(self, a):
        a.left = self.RR(a.left)
        return self.LL(a)

    def RL(self, a):
        a.right = self.LL(a.right)
        return self.RR(a)

    def insert1(self, a, data):  # avl树的插入过程
        if a is None:
            a = tree(data)
            return a
        else:
            if data < a.data:
                a.left = self.insert1(a.left, data)
            else:
                a.right = self.insert1(a.right, data)
            if self.deep(a.left) - self.deep(a.right) > 1:
                if data < a.left.data:  # LL型,即插入的数字在a节点的左边的左边
                    a = self.LL(a)
                else:  # LR型,即插入的数字在a节点的左边的右边
                    a = self.LR(a)
            elif self.deep(a.right) - self.deep(a.left) > 1:
                if data > a.right.data:  # RR型,即插入的数字在a节点的右边的右边
                    a = self.RR(a)
                else:  # RL型,即插入的数字在a节点的右边的左边
                    a = self.RL(a)
        return a


if __name__ == '__main__':
    login()