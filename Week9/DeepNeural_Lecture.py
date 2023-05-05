import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import SGDClassifier
from tensorflow import keras

# Fashion MNIST
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)

# 입력과 타깃 샘플
fig, axs = plt.subplots(1, 10, figsize=(10, 10))

for i in range(10):
	axs[i].imshow(train_input[i], cmap='gray_r')
	axs[i].axis('off')
# plt.show()

print()
print(">> [train_target[i] for i in range(10)]")
print([train_target[i] for i in range(10)])

print()
print(">> np.unique(train_target, return_counts=True)")
print(np.unique(train_target, return_counts=True))

# 로지스틱 회귀
train_scaled = train_input / 255.0		# 픽셀 값을 0 ~ 1 사이로 나타내기 위함 
train_scaled = train_scaled.reshape(-1, 28 * 28)

print()
print(">> train_scaled.shape")
print(train_scaled.shape)			# (28, 28) ==> (784)개로 flatten

# sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)			# 회귀를 위한 SGDClassifier 사용 중

# scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print()
print(">> np.mean(scores['test_score'])")
# print(np.mean(scores['test_score']))

# 텐서플로우와 케라스
# 케라스 모델 만들기
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

print()
print(">> train_scaled.shape, train_target.shape")
print(train_scaled.shape, train_target.shape)

print()
print(">> val_scaled.shape, val_target.shape")
print(val_scaled.shape, val_target.shape)

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))			# Dense = 뉴런의 입력과 출력을 연결해주는 역할

model = keras.Sequential(dense)

# 모델 설정 및 훈련
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')			# 이진 분류(binary_crossentropy), 다중 분류(categorical_crossentropy)

print()
print(">> train_target[:10]")
print(train_target[:10])

model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)
