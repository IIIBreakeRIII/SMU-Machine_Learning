import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

input_data = pd.read_excel('./Fuel_Power.xlsx', header=None)
input_target = pd.read_excel('./Electric_Power.xlsx', header=None)
input_data = input_data.drop(columns=[0])

print(len(input_data))
print(len(input_target))

input_data = np.array(input_data)
input_target = np.array(input_target)

train_input, test_input, train_target, test_target = train_test_split(input_data, input_target, random_state=42)

train_input = train_input.reshape(-1, 2)
test_input = test_input.reshape(-1, 2)
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std

lr = LinearRegression()
lr.fit(train_input, train_target)

print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))

poly = PolynomialFeatures()
poly.fit(train_input)

train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

lr.fit(train_poly, train_target)

print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

ss = StandardScaler()
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

train_score_ridge = []
test_score_ridge = []
train_score_lasso = []
test_score_lasso = []

for alpha in alpha_list:
	ridge = Ridge(alpha=alpha)
	ridge.fit(train_scaled, train_target)
	train_score_ridge.append(ridge.score(train_scaled, train_target))
	test_score_ridge.append(ridge.score(test_scaled, test_target))
	lasso = Lasso(alpha=alpha)
	lasso.fit(train_scaled, train_target)
	train_score_lasso.append(lasso.score(train_scaled, train_target))
	test_score_lasso.append(lasso.score(test_scaled, test_target))

print(train_score_ridge)
print(test_score_ridge)
