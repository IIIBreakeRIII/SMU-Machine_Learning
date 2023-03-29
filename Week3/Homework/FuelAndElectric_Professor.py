import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ep = np.array(pd.read_excel('./Electric_Power.xlsx'))
fp = np.array(pd.read_excel('./Fuel_Power.xlsx'))

plt.plot(fp[:, 0])
plt.plot(fp[:, 1])
plt.plot(fp[:, 2])
plt.plot(ep * 10000)
plt.show()

fp = fp[:, 1:3]

poly = PolynomialFeatures(degree = 3, include_bias = False)
poly.fit(fp)
train_poly = poly.transform(fp)

tr_in, ts_in, tr_out, ts_out = train_test_split(train_poly, ep, test_size=0.5, random_state=42)

lr = LinearRegression()
lr.fit(tr_in, tr_out)

print(lr.score(tr_in, tr_out))
print(lr.score(ts_in, ts_out))
