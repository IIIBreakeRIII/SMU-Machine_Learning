import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# pandas로 불러오기
inputFuel = pd.read_excel('Fuel_Power.xlsx', engine='openpyxl', header=None)							# 입력 데이터, header=None 은 열제목이 없는 경우를 가정
inputElectric = pd.read_excel('Electric_Power.xlsx', engine='openpyxl', header=None)			# 타겟 데이터

# 입력 데이터와 타켓 데이터의 차이
# 회귀 분석 : 연속적인 출력값을 예측하는 것
# 회귀는 예측 변수(Predictor Variable, 설명변수(Explanatory Variable), 입력(Input))와
# 연속적인 반응 변수(Response Variable, 출력(Output), 타깃(Target))가
# 주어졌을 때 두 변수 사이의 관계를 찾음
# 즉, 우리가 흔히 얘기하는 y = ax + b의 형태를 갖는 것

# 필요없는 데이터 삭제
inputFuel = inputFuel.drop(columns=[0])

# 데이터 확인
# print(inputFuel)
# print(inputElectric)

# 현재 데이터 출력
inputFuel.plot()
inputElectric.plot()

plt.show()

# 데이터 형태 변경
inputFuel = np.array(inputFuel)
inputElectric = np.array(inputElectric)

# 데이터 나누기
train_input, test_input, train_target, test_target = train_test_split(inputFuel, inputElectric, random_state=42)

# 2차원 배열로 표시
# 이유 : sklearn에서는 벡터와 행렬을 나타내는 방법으로 numpy 배열을 표준으로 함.
# 따라서 x와 y를 각각 numpy.array로 변환해야함.
# 하나의 속성(feature)에 여러가지 값(sample)을 갖는 경우, reshape(-1, 1)을 적용하여 열벡터로 만들어야 함.
# X는 하나의 종속 변수 Y에 대한 여러가지 값을 가지기 때문
# 여기서 reshape(-1, 1)의 -1의 의미는
# 남은 열의 값은 특정 정수로 지정이 되었을 때,
# 남은 배열의 길이와 남은 차원으로부터 추정해서 알아서 하라는 뜻
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# 평균과 표준편차 구하기
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean)
test_scaled = (test_input - mean)

plt.scatter(train_input, train_target)
plt.show()

# 확인용
# print("---------- Mean Value ----------")
# print(mean)
# print("---------- std Value ----------")
# print(std)
# print("---------- Train Scaled ----------")
# print(train_scaled)
# print("---------- Test Scaled ----------")
# print(test_scaled)

lr = LinearRegression()
lr.fit(train_input, train_target)

# lr.coef : 기울기
# lr.intercept : 절편
# _ : ML에서 유도된 결과를 나타내는 기호
print("---------- coef, intercept ----------")
print(lr.coef_, lr.intercept_)
train_scaled.plot()
test_scaled.plot()
plt.show()

print("SCORE")
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))
