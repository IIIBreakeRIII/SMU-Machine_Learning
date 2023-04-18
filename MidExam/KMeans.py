import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
print(fruits.shape)

apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)

print(apple.shape)

# 샘플 평균의 히스토그램 - 픽셀값 분석
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()

# 픽셀 평균의 히스토그램
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()

# 평균 이미지 그리기
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()

# 평균과 가까운 사진 고르기
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))
print(abs_mean.shape)

apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
	for j in range(10):
		axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
		axs[i, j].axis('off')
plt.show()

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

print(km.labels_)

print(np.unique(km.labels_, return_counts=True))

def draw_fruits(arr, ratio=1):
	n = len(arr)	# n은 샘플 개수
	# 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산
	rows = int(np.ceil(n / 10))
	# 행이 1개이면 열 개수는 샘플 개수. 그렇지 않으면 10개
	cols = n if rows < 2 else 10
	fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)

	for i in range(rows):
		for j in range(cols):
			if i*10 + j < n:
				axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
			axs[i, j].axis('off')
	plt.show()

draw_fruits(fruits[km.labels_==0])

draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)

print(km.transform(fruits_2d[100:101]))

print(km.predict(fruits_2d[100:101]))

draw_fruits(fruits[100:101])

print(km.n_iter_)

# 최적의 K 찾기(클러스의 수 정하기)
inertia = []				# interia : 군집 내 샘플들과 군집 중심점의 거리 차이 --> 값이 적다는 것은 중심점 주변에 밀집해 있다는 뜻
for k in range(2, 7):
	km = KMeans(n_clusters=k, random_state=42)
	km.fit(fruits_2d)
	inertia.append(km.inertia_)

plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()

pca = PCA(n_components=50)
pca.fit(fruits_2d)

print(pca.components_.shape)

draw_fruits(pca.components_.rehshape(-1, 100, 100))

print(fruits_2d.shape)

fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
