def Findways(cur, aim, rest, tol, dp):
    if rest == 0:
        return 1 if cur == aim else 0
    """当前位置不是目标位置"""
    if cur == 1:
        ans = Findways(cur + 1, aim, rest - 1, tol, dp)
    elif cur == tol:
        ans = Findways(cur - 1, aim, rest - 1, tol, dp)
    else:
        ans = Findways(cur + 1, aim, rest - 1, tol, dp) + Findways(cur - 1, aim, rest - 1, tol, dp)
    dp[cur][rest] = ans
    return dp


if __name__ == "__main__":
    a = [1, 2, 3, 4, 5, 6, 7]
    tol = len(a)
    start = int(input("初始地方："))
    aim = int(input("目标地："))
    rest = int(input("多少步："))
    dp = [[-1] * (rest + 1)] * (tol + 1)  # dp表中的所有元素值为-1，这样等后期查询的时候，只要不为-1就说明这个地方算过，直接返回值而不用再递归
    for i in Findways(start, aim, rest, len(a), dp):
        print(len(i))
