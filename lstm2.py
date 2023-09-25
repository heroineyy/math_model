import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

# 加载数据
data = pd.read_csv('data/2.csv')

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['A','B','C','D','E','F','G','H','I','J','K']].values)

# 设置窗口值和滞后值
window_size = 3
lags = 2

# 数据处理和准备
X, y = [], []
for i in range(len(data_scaled) - window_size - lags):
    X_window = data_scaled[i:i + window_size]
    X_lagged = [data_scaled[i + j] for j in range(lags)]
    X_sample = np.concatenate((X_window, X_lagged), axis=0)
    X.append(X_sample)
    y.append(data_scaled[i + window_size + lags])

X, y = np.array(X), np.array(y)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 构建LSTM网络
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 11)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(11))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32)

# 测试
y_pred = model.predict(X_test)
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)

# 评估模型
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)
r2 = r2_score(y_test, y_pred)
print("R2:", r2)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", mape)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print("RMSE:", rmse)

# 可视化
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Values')
plt.show()
