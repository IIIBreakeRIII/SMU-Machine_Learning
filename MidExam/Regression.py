import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

# Data
data_size = 100
perch_length = np.random.randint(80, 440, (1, data_size)) / 10
perch_weight = perch_length ** 2 - 20 * perch_length + 110 + np.random.randn(1, data_size) * 50 + 100

# 데이터 확인
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(len(perch_length))
print(len(perch_weight))

train_input, test_input, train_target, test_target = train_test_split(perch_length.T, perch_weight.T, random_state=42)

train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print("knr.score")
print(knr.score(test_input, test_target))

test_prediction = knr.predict(test_input)
# 실제값과 예측값의 차이(Error)를 절대값으로 변환해 평균화
mae = mean_absolute_error(test_target, test_prediction)
print("mean_absolute_error")
print(mae)

print("knr.score - train")
print(knr.score(train_input, train_target))
print("knr.score - test")
print(knr.score(test_input, test_target))

# 이웃 개수를 조정하며 점수 조절 가능
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print("new knr.score - train")
print(knr.score(train_input, train_target))
print("new knr.socre - test")
print(knr.score(test_input, test_target))

# 학습 데이터 값 범위를 넘어서는 데이터

# 50cm 농어의 이웃 구하기
distances, indexes = knr.kneighbors([[50]])
# 훈련 세트의 산점도 그리기
plt.scatter(train_input, train_target)
# 훈련 세트 중에서 이웃 샘플만 다시 그림
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
# 50cm 농어 데이터
plt.scatter(50, 1033, marker='^')
plt.show()

# 선형회귀
lr = LinearRegression()
# 선형 회귀 모델 훈련
lr.fit(train_input, train_target)
# 50cm 농어에 대한 예측
print('lr.predict([[50]])')
print(lr.predict([[50]]))

print('lr.coef_, lr.intercept_')
print(lr.coef_, lr.intercept_)

# 선형 회귀 결과 확인하기
# 훈련 세트의 산점드를 그림
plt.scatter(train_input, train_target)
# 15에서 50까지 1차 방정식 그래프 그리기
# plt.plot([15, 50], [15 * lr.coef_ + lr.intercept_, 50 * lr.coef_ + lr.intercept_])
# 50cm 농어 데이터
plt.scatter(50, 1241.8, marker='^')
plt.show()

print('lr.score(train_input, train_target)')
print(lr.score(train_input, train_target))
print('lr.score(test_input, test_target)')
print(lr.score(test_input, test_target))

# 다항 회귀(2차)
# 2차 형태의 그래프를 그리기 위해서는 길이를 제곱한 항이 훈련 세트에 추가되어야 함
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

lr = LinearRegression()
lr.fit(train_poly, train_target)
print('lr.predict([[50**2, 50]])')
print(lr.predict([[50**2, 50]]))
print('lr.coef_, lr.intercept_')
print(lr.coef_, lr.intercept_)

# Weight = 1.01 * height ** 2 - 21.6 * height + 116.05

# 다항회귀의 결과를 직선으로 보기
# 구간별 직선을 그리기 위해 15에서 49까지의 정수 배열 만듦
point = np.arange(15, 50)

# 훈련 세트의 산점도를 그림
plt.scatter(train_input, train_target)

# 15에서 49까지 2차 방정식 그래프 그림
plt.plot(point, 1.10*point**2 - 19.6*point + 199.05)

# 50cm 농어 데이터
plt.scatter([50], [1574], marker='^')
plt.show()

print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
