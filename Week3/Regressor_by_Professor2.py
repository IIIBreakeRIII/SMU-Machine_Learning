import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

data_size = 10000000
perch_length = np.random.randint(80, 440, (1, data_size)) / 10																		#(1, 100)
perch_weight = perch_length ** 2 - 20 * perch_length + 110 + np.random.randn(1, data_size) * 50 	#(1, 100)

train_input, test_input, train_target, test_target = train_test_split(perch_length.T, perch_weight.T, random_state=42)

knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)

print(knr.predict([[50]])) 																						# 예측 잘 못함
distances, indexes = knr.kneighbors([[50]])
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
plt.scatter(50, 1033, marker='^')
plt.show()

print(np.mean(train_target[indexes]))

# 100짜리 농어 역시 예측 잘 못함
print(knr.predict([[100]]))
distances, indexes = knr.kneighbors([[100]])
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
plt.scatter(100, 1033, marker='^')
plt.show()

# 선형 회귀 - 직선 즉 1차항
lr = LinearRegression()
lr.fit(train_input, train_target)

print(lr.predict([[50]]))
print(lr.coef_, lr.intercept_)

plt.scatter(train_input, train_target)

# plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])

plt.scatter(50, 1241.8, marker='^')
plt.show()

print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))

# 다항 회귀 - 곡선 즉, 2차항 가능

train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

#%%

print(train_poly.shape, test_poly.shape)

#%%

lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50**2, 50]]))

#%%

print(lr.coef_, lr.intercept_)

#%%

# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다
point = np.arange(15, 50)

# 훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)

# 15에서 49까지 2차 방정식 그래프를 그립니다
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)

# 50cm 농어 데이터
plt.scatter([50], [1574], marker='^')
plt.show()

#%%

print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
