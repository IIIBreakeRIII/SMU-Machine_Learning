import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

# Data
data_size = 100
perch_length = np.random.randint(80,440, (1,data_size)) / 10 																							#(1, 100)
perch_weight = perch_length ** 2 - 20 * perch_length + 110 + np.random.randn(1, data_size) * 50 + 100			#(1, 100)

# Result 1
plt.scatter(perch_length, perch_weight)
plt.xlabel("Length")
plt.ylabel("Weight")
plt.show()

train_input, test_input, train_target, test_target = train_test_split(perch_length.T, perch_weight.T, random_state=42)				# T for Transform

train_input = train_input.reshape(-1, 1)						# 행, 렬 을 의미. -1은 디바이스에서 마음대로 설정해라, 열은 1열로 만들어라
test_input = test_input.reshape(-1, 1)

knr = KNeighborsRegressor()
knr.fit(train_input, train_target)

print(knr.score(test_input, test_target))						# 출력값 = 0.992... , R^2 = 1 - (target - predict)^2 의 합 / (target - average)^2 의 합

test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print("오류 절대값:", mae)																					# 19.157...
