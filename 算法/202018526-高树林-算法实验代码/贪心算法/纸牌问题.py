def Findways(poker,n):
    ave = sum(poker) // len(poker)
    cnt = 0
    for i in range(1, n):
        if poker[i] == ave:
            continue
        poker[i + 1] = poker[i + 1] + poker[i] - ave
        cnt += 1
    return cnt


if __name__ == '__main__':
    n = int(input('输入几堆纸牌：'))
    poker = list(map(int,input('输入每堆纸牌数：').split(' ')))+[0]*100
    print('要使纸牌一样多，利用贪心算法移动次数最少为%d次'%Findways(poker,n))