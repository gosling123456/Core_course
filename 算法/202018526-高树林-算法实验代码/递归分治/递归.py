def FindNum(lst, num, first, end):
    global mid
    if lst[0] > num:  # 考虑查找的数在左边界外侧的情况
        return "大于该数的最小元素位置为0"
    if lst[-1] < num:  # 考虑查找的数在右边界外侧的情况
        return "小于该数的最大元素位置为%d" % (len(lst) - 1)
    if first > end:  # 考虑未找到，但是待查找数在中间情况
        return "该数不在序列中，大于该数的最小元素位置为%d" % mid + "小于该数的最大元素位置为%d" % (mid - 1)
    mid = (first + end) // 2
    if lst[mid] == num:  # 找到的情况
        return "已找到，该数的下标为%d" % mid
    elif lst[mid] < num:  # 查找的元素在右侧
        first = mid + 1  # 中间元素设为新的起始位置，从而达到二分的效果
        return FindNum(lst, num, first, end)
    elif lst[mid] > num:  # 查找的元素在左侧
        end = mid - 1  # 中间元素设为新的起始位置，从而达到二分的效果
        return FindNum(lst, num, first, end)


if __name__ == '__main__':
    a = [1, 2, 3, 7, 8, 9]
    num = int(input("输入你要查找的数："))
    print(FindNum(a, num, 0, len(a) - 1))