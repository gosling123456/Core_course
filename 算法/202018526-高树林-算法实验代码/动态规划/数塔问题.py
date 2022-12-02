import copy


def findways(m,dp):
    dp_new = copy.deepcopy(dp)  # 深拷贝，因为dp要后续修改，但是又会用到原来的值
    temp = -float('inf')  # 因为要求最大，因此初始值要设为负无穷
    for i in range(m - 1, -1, -1):
        for j in range(i + 1):
            temp = max(dp[i][j], dp[i][j + 1])  # 找到第i行相邻两个元素中最大的两个元素
            dp[i - 1][j] = temp + dp_new[i - 1][j]  # 将最大的元素和他的父节点相加，进入循环迭代，最后一次迭代出最大值
    j = 0
    print('路径为：%d'%dp_new[0][0],end='')
    for i in range(1, m):
        cur = dp[i-1][j] - dp_new[i-1][j]
        if cur == dp[i][j + 1]:
            j += 1
        print('->%d' % dp_new[i][j], end='')
    print('')
    return temp


if __name__ == '__main__':
    m = int(input('输入数塔深度：'))
    dp = [[0] * (m + 1) for _ in range(m + 1)]  # 防止第12行出现越界情况，因此在这里需要设置规模比原规模大1
    for i in range(m):
        a = list(map(int, input('第%d层：' % i).split(' ')))
        dp[i] = a + [0] * (m - len(a) + 1)
    print('最大值为：',findways(m,dp))