import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 加载多变量时间序列数据
data = pd.read_csv('data/2.csv')  # 假设数据文件包含多个变量，每列代表一个变量

# 归一化每个变量
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values)


# 定义灰色预测函数
def grey_model(data):
    n = len(data)
    x0 = data[0, :]
    x1 = np.cumsum(data, axis=0)

    z1 = (x1[:-1, :] + x1[1:, :]) / 2.0

    B = np.vstack((-z1.T, np.ones(n - 1))).T
    Y = data[1:, :].T

    beta = np.linalg.lstsq(B, Y, rcond=None)[0]

    a, b = beta[0, :], beta[1, :]

    x_pred = np.zeros((n, data.shape[1]))
    x_pred[0, :] = x0
    for i in range(1, n):
        x_pred[i, :] = (x0 - b / a) * (1 - np.exp(a)) * np.exp(-a * (i - 1))

    return x_pred


# 划分已知和未来时间段
known_data = scaled_data[:, :]  # 已知部分
future_years = 40  # 未来时间段的长度

# 预测未来数据
predicted_data = grey_model(known_data)

# 反归一化预测数据
predicted_data = scaler.inverse_transform(predicted_data)

# 打印预测结果
print("已知数据：")
print(data[:])  # 已知数据
print("预测未来数据：")
print(predicted_data[-future_years:])  # 预测数据

# 画图
plt.figure(figsize=(12, 6))
for i in range(data.shape[1]):
    plt.plot(np.arange(len(data)), data.iloc[:, i], label=f'Actual Variable {i + 1}')
    plt.plot(np.arange(len(known_data), len(data)), predicted_data[:, i], label=f'Predicted Variable {i + 1}',
             linestyle='--')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Values')
plt.show()
