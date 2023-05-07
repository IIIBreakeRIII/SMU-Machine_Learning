import numpy as np

from tensorflow import keras
from sklearn.model_selection import train_test_split

data_size = 100000
train_data = np.random.randint(1, 1000, size=(data_size, 2))
train_ans = np.log(train_data[:, 0] / train_data[:, 1]).reshape(-1, 1)

model = keras.models.Sequential()
model.add(keras.layers.Dense(20, activation='relu'))
model.add(keras.layers.Dense(40, activation='relu'))
model.add(keras.layers.Dense(10, activation='elu'))
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(optimizer=keras.optimizers.Adam(0.00029), loss='mse', metrics='accuracy')
model.fit(train_data, train_ans, batch_size=50, epochs=20)

quest = np.array([100, 2]).reshape(1, 2)
quest_log = model.predict(quest, batch_size=1)
ans = np.exp(quest_log)

print(">> 100 / 2")
print(ans[0][0])
