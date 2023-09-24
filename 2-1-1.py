import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# 加载数据
data = pd.read_csv('data/after_dataclean.csv')  # 请替换为你的数据文件路径
years = data['year'].values.reshape(-1, 1)
GDP = data['gdp'].values.reshape(-1, 1)
Population = data['population'].values.reshape(-1, 1)
Energy_Consumption = data['energy'].values.reshape(-1, 1)

# 数据归一化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(np.hstack([years, GDP, Population, Energy_Consumption]))

# 划分训练集和测试集
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# 构建ARIMA模型
model = SARIMAX(train_data, order=(1, 1, 0))  # p=5, d=1, q=0，根据实际情况调整参数
model_fit = model.fit()

# 预测未来40年的GDP、人口和能源消耗量之和
predictions = model_fit.forecast(steps=40)[0]  # 预测未来40年的数据，steps参数表示预测的步数
predictions = scaler.inverse_transform(predictions)  # 将预测结果转换回原始数据的范围