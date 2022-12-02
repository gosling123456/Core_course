def Findways(n, a):
    for i in range(0, n):  # 将左端点排好序
        for j in range(n - 1 - i):
            if a[0][j] > a[0][j + 1]:
                temp = a[1][j]
                a[1][j] = a[1][j + 1]
                a[1][j + 1] = temp

                temp = a[0][j]
                a[0][j] = a[0][j + 1]
                a[0][j + 1] = temp
    length = a[1][0] - a[0][0]
    point = [0, 0]
    point[0] = a[0][0]
    point[1] = a[1][0]
    for i in range(n):
        if a[0][i] >= point[1]:
            temp = (a[1][i] - a[0][i])
            length += temp
            point[0] = a[0][i]
            point[1] = a[1][i]
        if a[0][i] < point[1]:
            if a[1][i] > point[1]:
                tmp = (a[1][i] - point[1])
                length += tmp
                point[0] = a[0][i]
                point[1] = a[1][i]
    return length


if __name__ == '__main__':
    n = int(input('一共有几条线段(n<100)：'))
    a = [[0] * n for _ in range(2)]
    for i in range(n):
        a[0][i], a[1][i] = input("第%d个点坐标x,y:" % (i + 1)).split(' ')
        a[0][i] = int(a[0][i])
        a[1][i] = int(a[1][i])
    print(Findways(n, a))