money = int(input("输入你的钱数："))
# money = 27
f = [0] * money
f[0] = 0
for i in range(money):
    f[i] = min(f[i - 2] + 1, f[i - 5] + 1, f[i - 7] + 1)
print(f[money - 1])
