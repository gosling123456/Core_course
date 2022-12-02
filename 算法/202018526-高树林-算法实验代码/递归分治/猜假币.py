import random


def find_fake_coins(coins, first, end):
    mid = (first + end) // 2
    if 0 == coins[mid]:
        return "找到了!假币为下标从0开始的第%d个" % mid
    if 0 in coins[mid + 1:]:
        first = mid + 1
        mid = (first + end) // 2
        return find_fake_coins(coins, first, end)
    if 0 in coins[first:mid]:
        end = mid - 1
        mid = mid = (first + end) // 2
        return find_fake_coins(coins, first, end)
    return "No Fake Coins!"


if __name__ == "__main__":
    print("=====================================================================================")
    print('                                   假币用0表示')
    print("=====================================================================================")
    num = int(input("硬币个数："))
    coins = [1] * num  # 1表示真币
    # a = random.randint(0, len(coins) - 1)
    # print("下标从0开始的第%d个为假币" % a)
    # coins[a] = 0  # 0表示假币
    coins[random.randint(0, len(coins) - 1)] = 0
    print("硬币序列：",coins)
    print(find_fake_coins(coins, 0, len(coins) - 1))