import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GM1NModel:
    def __init__(self, data):
        self.data = data

    def fit(self):
        X = self.data.values
        X1 = np.cumsum(X)
        Z1 = (X1[:-1] + X1[1:]) / 2.0
        B = np.vstack((-Z1, np.ones(len(Z1)))).T
        Y = X[1:]
        self.coefficients = np.linalg.lstsq(B, Y, rcond=None)[0]

    def predict(self, n):
        a, b = self.coefficients
        X = self.data.values
        X1 = np.cumsum(X)
        prediction = [(X[0] - b / a) * np.exp(-a * i) * (1 - np.exp(a)) for i in range(len(X), len(X) + n)]
        return prediction

if __name__ == "__main__":
    # 准备数据，示例数据为10年的能源消耗量
    data = pd.DataFrame({
        'Year': range(2012, 2022),
        'EnergyConsumption': [30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
    })

    # 创建GM(1, N)模型对象并拟合数据
    model = GM1NModel(data['EnergyConsumption'])
    model.fit()

    # 预测未来40年的能源消耗量
    n = 40
    predictions = model.predict(n)

    # 将预测结果添加到数据框
    future_years = range(2022, 2062)
    future_data = pd.DataFrame({'Year': future_years, 'EnergyConsumption': predictions})

    # 可视化历史数据和预测结果
    plt.figure(figsize=(10, 6))
    plt.plot(data['Year'], data['EnergyConsumption'], marker='o', label='Historical Data')
    plt.plot(future_data['Year'], future_data['EnergyConsumption'], marker='x', label='Predicted Data')
    plt.xlabel('Year')
    plt.ylabel('Energy Consumption')
    plt.title('Energy Consumption Forecast')
    plt.legend()
    plt.grid(True)
    plt.show()
