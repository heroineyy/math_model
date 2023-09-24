import pandas as pd
from sklearn.linear_model import LinearRegression

# 假设你有一个包含多个指标和对应数据的DataFrame，命名为df
# 其中，每一行代表一个观测值，每一列代表一个指标

# 提取自变量和因变量数据
X = df.drop('target_variable', axis=1)  # 自变量数据，不包括目标变量列
y = df['target_variable']  # 因变量数据，即目标变量列

# 创建多元线性回归模型对象
model = LinearRegression()

# 拟合模型
model.fit(X, y)

# 打印模型系数和截距
print("模型系数：", model.coef_)
print("截距：", model.intercept_)
