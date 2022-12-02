def Findways(n, m, c, g, dp):
    for i in range(1, m + 1):
        for j in range(40 * abs(max(c)) * max(g)):
            if dp[i - 1][j]:  # 在表中可自取
                for k in range(1, n + 1):
                    dp[i][j + c[k] * g[i]] = dp[i][j + c[k] * g[i]] + dp[i - 1][j]
    return dp[m][20 * abs(max(c)) * max(g)]


if __name__ == "__main__":
    print("说明：本题中两端最大位置均为输入为位置绝对值的最大值，挂钩数任意，钩码质量任意，最大值为输入的最大值")
    n = int(input("挂钩数n:"))
    m = int(input("钩码数m:"))
    c = [0] + list(map(int, input("输入位置").split(' '))) + [0] * (19 - n)
    g = [0] + list(map(int, input("输入重量").split(' ')))
    g = g + [0] * (max(g) - m)
    dp = [[0] * 500 * abs(max(c)) * max(g) for _ in range(25)]
    dp[0][20 * abs(max(c)) * max(g)] = 1
    print(Findways(n, m, c, g, dp))

# https://blog.csdn.net/csyifanZhang/article/details/106925190?spm=1001.2101.3001.6650.3&depth_1-utm_relevant_index=4
