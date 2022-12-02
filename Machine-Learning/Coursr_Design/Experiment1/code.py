import pandas as pd

path = 'FIFA2022\matches_1930_2018.csv'
# 使用pandas读入
data = pd.read_csv(path)                    # 读取文件中所有数据
# 按列分离数据
x = data[['ImageID', 'label']]                   # 读取某两列
print(x)
y = data[['ImageID']]  # 读取某一列
print(y)
