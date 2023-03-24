import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# pandas로 불러오기
inputFuel = pd.read_excel('Fuel_Power.xlsx', engine='openpyxl', header=None)							# 입력 데이터, header=None 은 열제목이 없는 경우를 가정
inputElectric = pd.read_excel('Electric_Power.xlsx', engine='openpyxl', header=None)			# 타겟 데이터

print(len(inputFuel))
print(len(inputElectric))

# 데이터 나누기
train_input, test_input, train_target, test_target = train_test_split(inputFuel, inputElectric, random_state=158)

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std

print(len(train_input))
print(len(test_input))
print(len(train_target))
print(len(test_target))

# plt.scatter(inputFuel.iloc[:, 0], inputFuel.iloc[:, 1], inputFuel.iloc[:, 2])							# iloc == 행, 열을 선택하는 함수
# plt.show()
