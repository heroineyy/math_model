import pandas as pd
from statsmodels.tsa.stattools import acf

# 加载数据
data = pd.read_csv('data/after_dataclean.csv')  # 请替换为你的数据文件路径
years = data['year'].values.reshape(-1, 1)
GDP = data['gdp'].values.reshape(-1, 1)
Population = data['population'].values.reshape(-1, 1)
Energy_Consumption = data['energy'].values.reshape(-1, 1)

# 计算自相关图（ACF）
acf_gdp = acf(GDP, nlags=10, fft=False)
acf_pop = acf(Population, nlags=10, fft=False)
acf_energy = acf(Energy_Consumption, nlags=10, fft=False)

# 绘制自相关图（ACF）
import matplotlib.pyplot as plt

plt.figure(figsize=(11, 8))
plt.subplot(311)
plt.plot(range(11), acf_gdp)
plt.title('GDP')
plt.subplot(312)
plt.plot(range(11), acf_pop)
plt.title('Population')
plt.subplot(313)
plt.plot(range(11), acf_energy)
plt.title('Energy Consumption')
plt.tight_layout()
plt.show()
