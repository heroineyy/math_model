import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load data
data = pd.read_csv('data/after_dataclean.csv')  # 请替换为你的数据文件路径

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['population', 'gdp', 'energy']].values)

window_size = 5
X, y = [], []
for i in range(len(data_scaled) - window_size):
    X.append(data_scaled[i:i + window_size])
    y.append(data_scaled[i + window_size, :])

X, y = np.array(X), np.array(y)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Construct LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 3)))
model.add(LSTM(units=50))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Predict on test set
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print("RMSE:", rmse)


# Predict future values
future_predictions = []
# 例如根据2017年到2020年的数据去预测2021年
last_window = X_test[-1]
for i in range(40):
    next_value = model.predict(last_window.reshape(1, window_size, 3))
    future_predictions.append(scaler.inverse_transform(next_value))
    last_window = np.append(last_window, next_value,axis= 0)[1:]


# 绘图
# 将往年的energy和预测的进行拼接
pre_energy = []
for i in range(40):
    pre_energy=np.append(pre_energy,future_predictions[i][0][2])
before_energy=np.array(data[['energy']].values).ravel()
all_energy = np.concatenate((before_energy,pre_energy),axis=0)

pre_population = []
for i in range(40):
    pre_population=np.append(pre_population,future_predictions[i][0][0])
before_population=np.array(data[['population']].values).ravel()
all_population = np.concatenate((before_population,pre_population),axis=0)


pre_gdp = []
for i in range(40):
    pre_gdp=np.append(pre_gdp,future_predictions[i][0][1])
before_gdp=np.array(data[['gdp']].values).ravel()
all_gdp = np.concatenate((before_gdp,pre_gdp),axis=0)

plt.plot(all_energy,'r-',label="energy")
plt.plot(all_gdp,'g-',label="gdp")
plt.plot(all_population,'b-',label="population")
#todo:改变x轴的值，但是y轴的相对位置不变
plt.legend()
plt.show()

# xpoints = np.array([2021,2022,2023,2024,2025])
# pre_145=pre[0:5]
# pre_215=pre[-5:]




