import random

import tensorflow as tf
import numpy as np

from tensorflow import keras
from sklearn.model_selection import train_test_split

num1 = []
num2 = []
ans = []

for _ in range(100000):
	num1.append(random.randint(1, 100000))
	num2.append(random.randint(1, 100000))

for i in range(100000):
	ans.append(num1[i] + num2[i])

input_list = np.column_stack([num1, num2])
target_list = np.array(ans).reshape(100000, 1)

train_scaled, val_scaled, train_target, val_target = train_test_split(input_list, ans, test_size=0.2, random_state=42)

print(train_scaled.shape)
print(train_target.shape)
