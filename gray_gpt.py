import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 原始数据，9个指标，每个指标10年的数据
# 加载数据
data = pd.read_csv('data/2.csv')

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['A','B','C','D','E','F','G','H','I','J','K']].values)
# 数据预处理
t = np.arange(2010, 2020)  # 假设从2000年开始，到2009年结束
y = data_scaled.T  # 转置数据，使每一行代表一个指标，每列代表一年

# 灰色模型建立
def gm11(x0):
    x1 = np.cumsum(x0)
    z1 = (x1[:-1] + x1[1:]) / 2.0
    z1 = z1.reshape((len(z1), 1))
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Y = x0[1:].reshape((len(x0)-1, 1))
    a, b = np.dot(np.linalg.inv(np.dot(B.T, B)), np.dot(B.T, Y))
    x_pred = np.zeros(len(x0))
    x_pred[0] = x0[0]
    for i in range(1, len(x0)):
        x_pred[i] = (x0[0] - b / a) * (1 - np.exp(a)) * np.exp(-a * (i - 1))
    return x_pred

# 进行灰色预测
y_pred = np.zeros_like(y)
for i in range(len(y)):
    y_pred[i] = gm11(y[i])

# 可视化预测结果
plt.figure(figsize=(12, 6))
for i in range(len(y)):
    plt.plot(t, y[i], marker='o', label=f'Indicator {i+1} (Original)')
    plt.plot(t, y_pred[i], linestyle='--', label=f'Indicator {i+1} (Predicted)')
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Grey Prediction for Future 40 Years')
plt.legend()
plt.grid(True)
plt.show()
