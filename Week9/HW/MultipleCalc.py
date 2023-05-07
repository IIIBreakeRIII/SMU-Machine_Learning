import numpy as np

from tensorflow import keras

data_size = 100000
train_data = np.random.randint(1, 100, size=(data_size, 2))
train_ans = np.log(train_data[:, 0] * train_data[:, 1]).reshape(-1, 1)

model = keras.models.Sequential()
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(25, activation='elu'))
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.0001), metrics='accuracy')
model.fit(train_data, train_ans, batch_size=50, epochs=30)

quest = np.array([20, 20]).reshape(1, 2)
quest_log = model.predict(quest, batch_size=1)
ans = np.exp(quest_log)

print(">> 20 * 20")
print(ans[0][0])
