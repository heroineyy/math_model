import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

# ��������
data = pd.read_csv('data/2.csv')
# ��һ��
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['A','B','C','D','E','F','G','H','I','J','K']].values)

# ���ô���ֵͨ�����ڵ�ֵ�ƶ�δ��ֵ
window_size = 5
X, y = [], []
for i in range(len(data_scaled) - window_size):
    X.append(data_scaled[i:i + window_size])
    y.append(data_scaled[i + window_size, :])

# ����ֵ����Ϊndarray
X, y = np.array(X), np.array(y)

# ����ѵ�����Ͳ��Լ�
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# ����LSTM����
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 11))) # windows_size*input_nums
model.add(LSTM(units=50))
model.add(Dense(11))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# ���ԣ�����RMSE��MAE��R2��MAPE
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)
print("MAPE:", mape)