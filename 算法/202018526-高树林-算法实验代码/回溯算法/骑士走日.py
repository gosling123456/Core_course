def FindPath(I, J):
    chess[I][J] = 1
    posCount = 0
    posI = [1, 1, 2, 2, -1, -1, -2, -2]
    posJ = [2, -2, 1, -1, 2, -2, 1, -1]
    nowI = I
    nowJ = J
    nexI = [0] * 8
    nexJ = [0] * 8
    for queuenumber in range(2, 65):
        posCount = 0
        for k in range(8):
            nextI = nowI + posI[k]
            nextJ = nowJ = posJ[k]
            if nextI >= 8 or nextI < 0 or nextJ >= 8 or nextJ < 0:
                continue
            if chess[nextI][nextJ] == 0:
                nexI[posCount] = nextI
                nexJ[posCount] = nextJ
                posCount += 1
        if posCount == 0 and queuenumber < 63:
            return False
        minPosCounter = 8
        for posNum in range(posCount):
            posCountTemp = 0
            for k in range(8):
                nextnextI = nexI[posNum] + posI[k]
                nextnextJ = nexJ[posNum] + posJ[k]
                if nextnextI >= 8 or nextnextI < 0 or nextnextJ >= 8 or nextnextJ < 0:
                    continue
                if chess[nextnextI][nextnextJ] == 0:
                    posCountTemp += 1
                if minPosCounter > posCountTemp:
                    minPosCounter = posCountTemp
                    nowI = nexI[posNum]
                    nowJ = nexJ[posNum]
        chess[nowI][nowJ] = queuenumber
    return True


if __name__ == "__main__":
    chess = [[0] * 8 for _ in range(8)]
    a, b = input('请输入起始点坐标（x y):').split(' ')
    a = int(a)
    b = int(b)
    if FindPath(a, b):
        print("路径为：")
        for i in range(8):
            for j in range(8):
                print("%4d" % chess[i][j])
            print('')
    else:
        print('未找到遍历所有结点的路径！')