# 데이터 로드
import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt

# 삼성전자
df = fdr.DataReader('005930', '2018-05-04', '2022-12-31')

# 데이터 전처리(0과 1 사이 값으로 변환)
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# 학습 데이터와 타겟 설정(종가)
dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfx = dfx[['Close']]
dfx = dfx[['Open', 'High', 'Low', 'Volume']]
dfx.describe()

# N일의 데이터를 종가로 연결
# 입력 데이터와 타겟을 설정
X = dfx.values.tolist()
y = dfx.values.tolist()

# 10일의 데이터를 종가로 연결
window_size = 10
data_X = []
data_y = []
for i in range(len(y) - window_size):
    _X = X[i : i + window_size]
    _y = y[i + window_size]
    data_X.append(_X)
    data_y.append(_y)
print(_X, "->", _y)
