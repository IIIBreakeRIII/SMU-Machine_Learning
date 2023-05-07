import random

import tensorflow as tf
import numpy as np

from tensorflow import keras
from sklearn.model_selection import train_test_split

num1 = []
num2 = []
ans = []

for _ in range(100000):
	num1.append(random.randint(1, 1000) / 100)
	num2.append(random.randint(1, 1000) / 100)

for i in range(100000):
	ans.append(num1[i] + num2[i])

input_list = np.column_stack([num1, num2])
target_list = np.array(ans).reshape(100000, 1)

train_scaled, val_scaled, train_target, val_target = train_test_split(input_list, target_list, test_size=0.2, random_state=42)

model = keras.Sequential()
model.add(keras.layers.Dense(50, input_dim=2, activation='relu'))
model.add(keras.layers.Dense(25, activation='relu'))
model.add(keras.layers.Dense(5, activation='relu'))
model.add(keras.layers.Dense(1, activation='relu'))

model.compile(optimizer='adam', loss='mse', metrics='accuracy')
model.fit(train_scaled, train_target, batch_size=50, epochs=20)

model.evaluate(val_scaled, val_target)

result1 = model.predict([[100, 200]])
print(">> 100 + 200")
print(result1[0])

result2 = model.predict([[203.91, 23.313]])
print(">> 203.91 + 23.313")
print(result2[0])
