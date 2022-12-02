def FindWays(cur, rest, aim, tot):  # cur:当前位置 rest:剩余位置 aim:木标位置 tot:总共位置
    if rest == 0:
        return 1 if cur == aim else 0  # 若当前位置为目标元素，则找到了一种方法，否则没找到
    else:
        if cur == 1:
            return FindWays(2, rest-1, aim, tot)
        elif cur == tot:
            return FindWays(tot - 1, rest-1, aim, tot)
        else:
            return FindWays(cur - 1, rest-1, aim, tot) + FindWays(cur + 1, rest-1, aim, tot)


if __name__ == '__main__':
    a = [10, 20, 30, 40, 50]
    start = int(input("初始地方："))
    aim = int(input("目标地："))
    rest = int(input("多少步："))
    print('共有',FindWays(start,rest,aim,len(a)),'种方法')
