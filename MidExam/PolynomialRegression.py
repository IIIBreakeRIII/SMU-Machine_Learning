import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0, 1000.0])
df = pd.read_csv('https://bit.ly/perch_csv')
perch_full = df.to_numpy()

train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

# 다항 특성 만들기
# degree = 2
poly = PolynomialFeatures()
poly.fit([[2, 3]])

print(poly.transform([[2, 3]]))

# LinearRegrssion 실행
poly = PolynomialFeatures(include_bias=False)

poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print("train_poly.shape")
print(train_poly.shape)
print("get_feature_names_out()")
poly.get_feature_names_out()

lr = LinearRegression()
lr.fit(train_poly, train_target)

print("lr.score(train_poly, train_target)")
print(lr.score(train_poly, train_target))
print("lr.score(test_poly, test_target)")
print(lr.score(test_poly, test_target))

# 더 많은 특성 만들기
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print("train_poly.shape")
print(train_poly.shape)
lr.fit(train_poly, train_target)
print("lr.score(train_poly, train_target)")
print(lr.score(train_poly, train_target))
print('lr.score(test_poly, test_target)')
print(lr.score(test_poly, test_target))

# 규제, Regularization
# 과도한 학습이 불가하도록 오차 발생
ss = StandardScaler()
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 릿지 회귀 : 데이터들의 각각의 제곱및 곱 등이 반영된 입력 데이터 만듦
ridge = Ridge()
ridge.fit(train_scaled, train_target)

print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

# 적절한 강도를 찾아서 넣어줌
