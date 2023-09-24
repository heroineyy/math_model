import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
# 读取数据
data = pd.read_csv('data/after_dataclean.csv')
# 将年份设为索引并将两列合并为一个时间序列
data['year'] = data['year'].astype(int)
data['population'] = data['population'].astype(float)
data['energy'] = data['energy'].astype(float)
data['gdp'] = data['gdp'].astype(float)
data_ts = pd.concat([data['year'], data[['population', 'energy', 'gdp']]], axis=1)
data_ts.set_index('year', inplace=True)
# 预测未来40年的人口、GDP和经济（能源消费量）变化
# 定义ARIMA模型，其中population和energy作为内生变量，分别使用AR(1)和MA(1)参数来捕捉它们之间的相关性；gdp作为外生变量，使用AR(1)参数来捕捉它与能源消费量之间的相关性
print('-----gdp------')
arima_model_gdp = ARIMA(data_ts['gdp'], order=(1, 1, 1))
arima_model_result_gdp = arima_model_gdp.fit()
forecast_result_gdp = arima_model_result_gdp.forecast(steps=40)
print(forecast_result_gdp)

print('-----population------')
arima_model_population = ARIMA(data_ts['population'], order=(1, 1, 1))
arima_model_result_population = arima_model_population.fit()
forecast_result_population = arima_model_result_population.forecast(steps=40)
print(forecast_result_population)

print('-----population------')
arima_model_energy = ARIMA(data_ts['energy'], order=(1, 1, 1))
arima_model_result_energy = arima_model_energy.fit()
forecast_result_energy = arima_model_result_energy.forecast(steps=40)
print(forecast_result_energy)



