import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# pandas로 불러오기
inputFuel = pd.read_excel('Fuel_Power.xlsx', engine='openpyxl', header=None)							# 입력 데이터, header=None 은 열제목이 없는 경우를 가정
inputElectric = pd.read_excel('Electric_Power.xlsx', engine='openpyxl', header=None)			# 타겟 데이터

# 필요없는 데이터 삭제
inputFuel = inputFuel.drop(columns=[0])

# 데이터 확인
print(inputFuel)
print(inputElectric)

# 현재 데이터 출력
inputFuel.plot()
inputElectric.plot()

plt.show()

# 데이터 나누기
train_input, test_input, train_target, test_target = train_test_split(inputFuel, inputElectric, random_state=158)


# 평균과 표준차 구하기
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean)
test_scaled = (test_input - mean)

# 확인용
print("---------- Mean Value ----------")
print(mean)
print("---------- std Value ----------")
print(std)
print("---------- Train Scaled ----------")
print(train_scaled)
print("---------- Test Scaled ----------")
print(test_scaled)

lr = LinearRegression()
lr.fit(train_input, train_target)

# lr.coef : 기울기
# lr.intercept : 절편
# _ : ML에서 유도된 결과를 나타내는 기호
print(lr.coef_, lr.intercept_)

train_scaled.plot()
test_scaled.plot()
plt.show()

print("SCORE")
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))
