import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout


df = fdr.DataReader('005930', '2018-05-04', '2023-05-24') #삼전
print(df.shape)

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
return numerator / (denominator + 1e-7)

dfx = df[['Open','High','Low','Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]
dfx.describe() #모든 값이 0과 1 사이인 것 확인

X = dfx.values.tolist()
y = dfy.values.tolist()

window_size = 10
data_X = []
data_y = []
for i in range(len(y) - window_size):
    _X = X[i : i + window_size]
    _y = y[i + window_size]
    data_X.append(_X)
    data_y.append(_y)
print(_X, "->", _y)

print('전체 데이터의 크기 :', len(data_X), len(data_y))
start = int(len(data_y) * 0.7)
train_X = np.array(data_X[start : (len(data_y)-1)])
train_y = np.array(data_y[start : len(data_y)])
train_y=((train_y[1:371]-train_y[0:370])>0)*1

print(train_y)

model = Sequential()
model.add(LSTM(units=20, activation='relu', return_sequences=True, input_shape=(10, 4)))
model.add(Dropout(0.1))
model.add(LSTM(units=20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_X, train_y, epochs=70, batch_size=10)

result=model.predict(data_X[(len(data_y)-1):(len(data_y)) ])
print(result)


'''
pred_y = model.predict(test_X)
plt.figure()
plt.plot(test_y, color='red', label='real SEC stock price')
plt.plot(pred_y, color='blue', label='predicted SEC stock price')
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()
print("내일 SEC 주가 :", df.Close[-1] * pred_y[-1] / dfy.Close[-1], 'KRW')
'''
