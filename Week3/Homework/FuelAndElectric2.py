import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 데이터 불러오기(Pandas)
input_data = pd.read_excel('Fuel_Power.xlsx', header=None)
input_target = pd.read_excel('Electric_Power.xlsx', header=None)

# 입력 데이터와 타켓 데이터의 차이
# 회귀 분석 : 연속적인 출력값을 예측하는 것
# 회귀는 예측 변수(Predictor Variable, 설명변수(Explanatory Variable), 입력(Input))와
# 연속적인 반응 변수(Response Variable, 출력(Output), 타깃(Target))가
# 주어졌을 때 두 변수 사이의 관계를 찾음
# 즉, 우리가 흔히 얘기하는 y = ax + b의 형태를 갖는 것

# 현재 데이터 분포도 확인
input_data.plot()
input_target.plot()
plt.show()

# 첫번째 열 제거(필요없는 데이터)
input_data = input_data.drop(columns=[0])

# 데이터 확인
print(input_data)
print(input_target)

# Array로 형변환
input_data = np.array(input_data)
input_target = np.array(input_target)

# 데이터 나누기
train_input, test_input, train_target, test_target = train_test_split(input_data, input_target, random_state=42)

# 나눠진 데이터 길이 확인
print(len(train_input))
print(len(train_target))
print(len(test_input))
print(len(test_target))

# 2차원 배열로 표시
# 이유 : sklearn에서는 벡터와 행렬을 나타내는 방법으로 numpy 배열을 표준으로 함.
# 따라서 x와 y를 각각 numpy.array로 변환해야함.
# 하나의 속성(feature)에 여러가지 값(sample)을 갖는 경우, reshape(-1, 1)을 적용하여 열벡터로 만들어야 함.
# X는 하나의 종속 변수 Y에 대한 여러가지 값을 가지기 때문
# 여기서 reshape(-1, 1)의 -1의 의미는
# 남은 열의 값은 특정 정수로 지정이 되었을 때,
# 남은 배열의 길이와 남은 차원으로부터 추정해서 알아서 하라는 뜻
train_input = train_input.reshape(-1, 2)
test_input = test_input.reshape(-1, 2)

# 1번째 열과 2번째 열의 차이를 보여줌
plt.scatter(train_input[:, 0], train_target[:, 0])
plt.scatter(train_input[:, 1], train_target[:, 0])
plt.show()

# 표준화
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std

# 선형회귀 정의
lr = LinearRegression()
lr.fit(train_input, train_target)

# 기울기와 절편 구하기
# lr.coef : 기울기
# lr.intercept : 절편
# _ : ML에서 유도된 결과를 나타내는 기호
print(lr.coef_, lr.intercept_)
print("coef =", lr.coef_)
print("intercept =", lr.intercept_)

# 현재 1차원 선형 데이터로 학습시켰을 때 학습 정확도와 테스트 정확도
print("--------------- 선형 회귀(1차) ---------------")
print("----------------- 학습 정확도 ----------------")
print(lr.score(train_input, train_target))
print("---------------- 테스트 정확도 ---------------")
print(lr.score(test_input, test_target))
print("----------------------------------------------")

# PolynomialFeatures = 다항회귀
# 데이터의 형태가 비선형일때, 데이터에 각 특성의 제곱을 추가하여
# 특성이 추가된 비선형 데이터를 선형 회귀 모델로 훈련시키는 방법
# 쉽게 생각하면, degree값(차수)의 변화에 따라 정확도가 달라지게 됨
# 기본 degree의 default = 2
# include_bias = 0차항의 유무
poly = PolynomialFeatures(degree=2, include_bias=False)

# 학습 모델을 학습
poly.fit(train_input)

# Polynomial.transform = 다항식의 특징을 생성
# 입력 데이터에 대해 주어진 차수까지의 다항식 함수를 적용한 후,
# 새로운 특징으로 반환
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

# train_poly.shape = (n_samples, n_features)와 같은 형태를 가지며
# train_poly에 포함된 행과 열의 개수 의미
# 118, 6이면 118개의 샘플데이터와 각 샘플은 6개의 특징을 가짐
print("--------------- train_poly.shape ---------------")
print(train_poly.shape)				# 118, 6
print("------------------------------------------------")

# 모든 다항식 특징의 이름을 반환
print("--------------- poly.get_feature_names_out() ---------------")
print(poly.get_feature_names_out())
print("------------------------------------------------------------")

# train_poly와 train_target을 학습
lr.fit(train_poly, train_target)

# 정확도 예측
print("--------------- 선형 회귀(2차, PolynomialFeatures) ---------------")
print("--------------------------- 학습 정확도 --------------------------")
print(lr.score(train_poly, train_target))
print("-------------------------- 테스트 정확도 -------------------------")
print(lr.score(test_poly, test_target))
print("------------------------------------------------------------------")

# Regularization
# StandardScaler = 주어진 데이터셋의 각 특징을 평균이 0이고
# 분산이 1인 표준정규분포로 변환해주는 기능 수행
# 즉 정규화(normalization)의 역할
ss = StandardScaler()
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# Ridge Model
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print("--------------- Ridge Model, 규제 강도 없음 ---------------")
print("----------------------- 학습 정확도 -----------------------")
print(ridge.score(train_scaled, train_target))
print("---------------------- 테스트 정확도 ----------------------")
print(ridge.score(test_scaled, test_target))
print("------------------------------------------------------------")

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

train_score_ridge = []
test_score_ridge = []
train_score_lasso = []
test_score_lasso = []

for alpha in alpha_list:
	ridge = Ridge(alpha = alpha)
	ridge.fit(train_scaled, train_target)
	train_score_ridge.append(ridge.score(train_scaled, train_target))
	test_score_ridge.append(ridge.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score_ridge)
plt.plot(np.log10(alpha_list), test_score_ridge)
plt.xlabel("Alpha Value")
plt.ylabel("R^2")
plt.show()

ridge = Ridge(alpha=1)
ridge.fit(train_scaled, train_target)

print("--------------- Ridge Model, 규제 강도 있음, alpha = 1 ---------------")
print("----------------------------- 학습 정확도 ----------------------------")
print(ridge.score(train_scaled, train_target))
print("---------------------------- 테스트 정확도 ---------------------------")
print(ridge.score(test_scaled, test_target))
print("-----------------------------------------------------------------------")

# Lasso Model
lasso = Lasso()
lasso.fit(train_scaled, train_target)

print("--------------- Lasso Model, 규제 강도 없음 ---------------")
print("----------------------- 학습 정확도 -----------------------")
print(lasso.score(train_scaled, train_target))
print("---------------------- 테스트 정확도 ----------------------")
print(lasso.score(test_scaled, test_target))
print("-----------------------------------------------------------")

for alpha in alpha_list:
	lasso = Lasso(alpha = alpha)
	lasso.fit(train_scaled, train_target)
	train_score_lasso.append(lasso.score(train_scaled, train_target))
	test_score_lasso.append(lasso.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score_lasso)
plt.plot(np.log10(alpha_list), test_score_lasso)
plt.xlabel("Alpha Value")
plt.ylabel("R^2")
plt.show()

lasso = Lasso(alpha=0.1)
lasso.fit(train_scaled, train_target)

print("--------------- Lasso Model, 규제 강도 있음, alpha = 0.1 ---------------")
print("------------------------------ 학습 정확도 -----------------------------")
print(lasso.score(train_scaled, train_target))
print("----------------------------- 테스트 정확도 ----------------------------")
print(lasso.score(test_scaled, test_target))
print("------------------------------------------------------------------------")
