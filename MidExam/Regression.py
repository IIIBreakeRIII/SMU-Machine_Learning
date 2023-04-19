import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

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


