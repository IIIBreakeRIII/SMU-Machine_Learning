import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

input_data = pd.read_excel('Fuel_Power.xlsx', header=None)
input_target = pd.read_excel('Electric_Power.xlsx', header=None)

input_data = input_data.drop(columns=[0])

input_data = np.array(input_data)
input_target = np.array(input_target)

train_input, test_input, train_target, test_target = train_test_split(input_data, input_target, random_state=42)

print(len(train_input))
print(len(train_target))
print(len(test_input))
print(len(test_target))

train_input = train_input.reshape(-1, 2)
test_input = test_input.reshape(-1, 2)

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std

lr = LinearRegression()
lr.fit(train_input, train_target)

print(lr.coef_, lr.intercept_)

# 1번째 열과 2번째 열의 차이를 보여줌
plt.scatter(train_input[:, 0], train_target[:, 0])
plt.scatter(train_input[:, 1], train_target[:, 0])
plt.show()

# PolynomialFeatures = 다항회귀
# 데이터의 형태가 비선형일때, 데이터에 각 특성의 제곱을 추가하여
# 특성이 추가된 비선형 데이터를 선형 회귀 모델로 훈련시키는 방법
poly = PolynomialFeatures()

train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
