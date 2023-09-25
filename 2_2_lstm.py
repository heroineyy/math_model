import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

# 加载数据
data = pd.read_csv('data/2.csv')

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['A','B','C','D','E','F','G','H','I','J','K']].values)

# 设置窗口值通过窗口的值推断未来值
window_size = 3
X, y = [], []
for i in range(len(data_scaled) - window_size):
    X.append(data_scaled[i:i + window_size])
    y.append(data_scaled[i + window_size, :])

# 将数值设置为ndarray
X, y = np.array(X), np.array(y)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 构建LSTM网络
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 11))) # windows_size*input_nums
model.add(LSTM(units=50))
model.add(Dense(11))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# 测试，
y_pred = model.predict(X_test)

y_pred=scaler.inverse_transform(y_pred)
y_test=scaler.inverse_transform(y_test)
print(y_pred)
print(y_test)

# 计算MAE
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

# 计算R2
r2 = r2_score(y_test, y_pred)
print("R2:", r2)

# 计算MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", mape)

# 计算RMSE
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print("RMSE:", rmse)




# # 画图？啥子效果？
# future_predictions = []
# # 预测
# for i in range(6):
#     next_value = model.predict(X[i].reshape(1, window_size, ))
#     future_predictions.append(scaler.inverse_transform(next_value))
#
# y = scaler.inverse_transform(y)
# plt.plot(future_predictions[:][-1],color='red')
# # plt.plot(y[:][-1],color='blue')
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.show()


#print(future_predictions)



# pre = []
# for i in range(40):
#     pre=np.append(pre,future_predictions[i][0][2])
# # 2021-2025
# print(pre[0:5])
# # 2056-2060
# print(pre[-5:])

# 画图
# future_data = pd.DataFrame({'Energy': pre})
# plt.plot(future_data)
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.show()
