import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import SGDClassifier

from tensorflow import keras

print(tf.config.list_physical_devices('GPU'))

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)

fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
	axs[i].imshow(train_input[i], cmap='gray_r')
	axs[i].axis("off")

plt.show()

print([train_target[i] for i in range(10)])

print(np.unique(train_target, return_counts=True))

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

print("train_scaled.shape")
print(train_scaled.shape)

sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)

scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print("np.mean(scores['test_score'])")
print(np.mean(scores['test_score']))

train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

print("train_scaled.shape, train_target.shape")
print(train_scaled.shape, train_target.shape)
print("val_scaled.shape, val_target.shape")
print(val_scaled.shape, val_target.shape)

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
model = keras.Sequential(dense)
