def Findways(cur, rest, aim, tol):
    dp = [[0] * (rest+1) for _ in range(tol+1)]
    dp[aim][0] = 1
    for i in range(1,rest+1):
        dp[1][i] = dp[2][i-1]
        for j in range(2,tol):
            dp[j][i] = dp[j-1][i-1] + dp[j+1][i-1]
        dp[tol][i] = dp[tol-1][i-1]
    return dp


if __name__ == '__main__':
    for i in Findways(2,6,4,5):
        for j in i:
            print(j,end='\t')
        print('')
