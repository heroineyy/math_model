from decimal import *
import numpy as np
import pandas as pd
from scipy import io, integrate, linalg, signal
from scipy.sparse.linalg import eigs
from scipy.integrate import odeint
# %mtplotlib-inline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
import sys  # 导入sys模块
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000

class GM11():
    def __init__(self):
        self.f = None

    def isUsable(self, X0):
        '''判断是否通过光滑检验'''
        X1 = X0.cumsum()
        rho = [X0[i] / X1[i - 1] for i in range(1, len(X0))]
        rho_ratio = [rho[i + 1] / rho[i] for i in range(len(rho) - 1)]
        print(rho, rho_ratio)
        flag = True
        for i in range(2, len(rho) - 1):
            if rho[i] > 0.5 or rho[i + 1] / rho[i] >= 1:
                flag = False
        if rho[-1] > 0.5:
            flag = False
        if flag:
            print("数据通过光滑校验")
            X1+=100
            self.isUsable(X1)
        else:
            print("该数据未通过光滑校验")

        '''判断是否通过级比检验'''
        lambds = [X0[i - 1] / X0[i] for i in range(1, len(X0))]
        X_min = np.e ** (-2 / (len(X0) + 1))
        X_max = np.e ** (2 / (len(X0) + 1))
        for lambd in lambds:
            if lambd < X_min or lambd > X_max:
                print('该数据未通过级比检验')
                # X0+=1000
                # self.isUsable(X0)
                return
        print('该数据通过级比检验')



    def train(self, X0):
        X1 = X0.cumsum(axis=0)  # [x_2^1,x_3^1,...,x_n^1,x_1^1] # 其中x_i^1为x_i^01次累加后的列向量
        Z = (-0.5 * (X1[:, -1][:-1] + X1[:, -1][1:])).reshape(-1, 1)
        # 数据矩阵(matrix) A、B
        A = (X0[:, -1][1:]).reshape(-1, 1)
        B = np.hstack((Z, X1[1:, :-1]))
        print('Z: ', Z.shape, 'B', B.shape, 'X1', X1.shape)
        # 求参数
        u = np.linalg.inv(np.matmul(B.T, B)).dot(B.T).dot(A)
        a = u[0][0]
        b = u[1:]
        print("灰参数a：", a, "，参数矩阵(matrix)b：", b.shape)
        self.f = lambda k, X1: (X0[0, -1] - (1 / a) * (X1[k]).dot(b)) * np.exp(-a * k) + (1 / a) * (X1[k]).dot(b)

    def predict(self, k, X0):
        '''
        :param k: k为预测的第k个值
        :param X0: X0为【k*n】的矩阵(matrix),n为特征的个数，k为样本的个数
        :return:
        '''
        X1 = X0.cumsum(axis=0)
        X1_hat = [float(self.f(k, X1)) for k in range(k)]
        X0_hat = np.diff(X1_hat)
        X0_hat = np.hstack((X1_hat[0], X0_hat))
        return X0_hat

    def evaluate(self, X0_hat, X0):
        '''
        根据后验差比及小误差概率判断预测结果
        :param X0_hat: 预测结果
        :return:
        '''
        S1 = np.std(X0, ddof=1)  # 原始数据样本标准差(standard deviation)
        S2 = np.std(X0 - X0_hat, ddof=1)  # 残差数据样本标准差(standard deviation)
        C = S2 / S1  # 后验差比
        Pe = np.mean(X0 - X0_hat)
        temp = np.abs((X0 - X0_hat - Pe)) < 0.6745 * S1
        p = np.count_nonzero(temp) / len(X0)  # 计算小误差概率
        print("原数据样本标准差(standard deviation)：", S1)
        print("残差样本标准差(standard deviation)：", S2)
        print("后验差：", C)
        print("小误差概率p：", p)


data = pd.read_csv("data/2.csv", encoding='ANSI')
# data.drop('供水总量', axis=1, inplace=True)
# 原始数据X
# 归一化每个变量
# scaler = MinMaxScaler()
# X = scaler.fit_transform(data.values)
X = data.values
# 训练集
X_train = X[:, :]
# 测试集
X_test = []

model = GM11()
model.isUsable(X_train[:, -1])  # 判断模型可行性
model.train(X_train)  # 训练
Y_pred = model.predict(len(X), X[:, :-1])  # 预测
Y_train_pred = Y_pred[:len(X_train)]
Y_test_pred = Y_pred[len(X_train):]
print(model.evaluate(Y_train_pred, X_train[:, -1]))  # 评估)
# score_test = model.evaluate(Y_test_pred, X_test[:, -1])

# Y_train_pred= scaler.inverse_transform(Y_train_pred)

# 可视化
plt.grid()
plt.plot(np.arange(len(Y_train_pred)), X_train[:, -1], '->')
plt.plot(np.arange(len(Y_train_pred)), Y_train_pred, '-o')
plt.legend(['负荷实际值', '灰色预测模型预测值'])
plt.title('训练集')
plt.show()

# # 可视化
# plt.grid()
# plt.plot(np.arange(len(Y_test_pred)), X_test[:, -1], '->')
# plt.plot(np.arange(len(Y_test_pred)), Y_test_pred, '-o')
# plt.legend(['负荷实际值', '灰色预测模型预测值'])
# plt.title('测试集')
# plt.show()




