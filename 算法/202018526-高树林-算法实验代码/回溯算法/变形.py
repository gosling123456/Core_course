def solve():
    queue = [sour]
    while len(queue):
        status_of_1 = queue[0]
        del queue[0]
        for i in range(4):
            for j in range(3):
                c1 = 1 << (i * 4 + j)
                c2 = 1 << (i * 4 + j + 1)
                if status_of_1 & c1 != status_of_1 & c2:
                    status_of_2 = status_of_1
                    status_of_2 ^= c1
                    status_of_2 ^= c2
                    if a[status_of_2] == -1:
                        a[status_of_2] = a[status_of_1] + 1
                        b[status_of_2] = (i * 4 + j) + 1
                        if status_of_2 == dest:
                            return True
                        queue.append(status_of_2)
        for i in range(3):
            for j in range(4):
                c1 = 1 << (i * 4 + j)
                c2 = 1 << (i * 4 + j + 4)
                if status_of_1 & c1 != status_of_1 & c2:
                    status_of_2 = status_of_1
                    status_of_2 ^= c1
                    status_of_2 ^= c2
                    if a[status_of_2] == -1:
                        a[status_of_2] = a[status_of_1] + 1
                        b[status_of_2] = - (i * 4 + j) - 4
                        if status_of_2 == dest:
                            return True
                        queue.append(status_of_2)
    return False


def output(status, moves):
    if status != sour:
        c1 = c2 = tem_state = 0
        if b[status] > 0:
            c1 = 1 << b[status] - 1
            c2 = 1 << b[status]
            status_of_temp = status
            status_of_temp ^= c1
            status_of_temp ^= c2
            output(status_of_temp, moves - 1)
            c1 = (b[status] - 1) // 4
            c2 = (b[status] - 1) % 4
            print("第%d步" % moves)
            map1[c1][c2], map1[c1][c2 + 1] = map1[c1][c2 + 1], map1[c1][c2]
            for i in range(4):
                for j in range(4):
                    print(map1[i][j], end=' ')
                print('')
        else:
            b[status] = -b[status]
            c1 = 1 << (b[status] - 4)
            c2 = 1 << b[status]
            status_of_temp = status
            status_of_temp ^= c1
            status_of_temp ^= c2
            output(status_of_temp, moves - 1)
            c1 = (b[status] - 4) // 4
            c2 = (b[status] - 4) % 4
            print("第%d步" % moves)
            map1[c1][c2], map1[c1][c2 + 1] = map1[c1][c2 + 1], map1[c1][c2]
            for i in range(4):
                for j in range(4):
                    print(map1[i][j], end='')
                print('')
            b[status] = -b[status]


if __name__ == '__main__':
    map1 = []
    Capacity = 1 << 16
    a = [-1] * Capacity
    b = [0] * Capacity
    print("请输入转换前的图形（0表示黄色方块，1表示黑色方块）：")
    end = []
    s = ''
    sour = dest = 0
    for i in range(4):
        s += input(' ')
    map1 = list(s)
    print(map1)
    for i in range(16):
        sour |= int(ord(map1[i]) - ord('0')) << i
    print(sour)
    print("请输入转换后的图形（0表示黄色方块，1表示黑色方块）：")
    s = ''
    for i in range(4):
        s += input(' ')
    map2 = list(s)
    for i in range(16):
        sour |= int(ord(map2[i]) - ord('0')) << i
    Capacity = 2 ** 16
    a = [-1] * Capacity
    b = [0] * Capacity
    solve()
    if a[dest] != -1:
        print("至少需要%d步" % a[dest])
        output(dest, a[dest])
