import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

LoadInput = np.load("./Week2-data_input.npy")
LoadTarget = np.load("./Week2-data_target.npy")

train_input, test_input, train_target, test_target = train_test_split(LoadInput, LoadTarget, stratify=LoadTarget, random_state = 1800)

kn = KNeighborsClassifier()

print("-------")
print(len(LoadInput))
print("-------")
print(len(LoadTarget))
print("-------")

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean)
test_scaled = (test_input - mean)
new = ([25, 500] - mean)

kn.fit(train_scaled, train_target)
print(kn.score(test_scaled, test_target))
print(kn.predict([new]))

distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:, 0], train_scaled[:, 1], s=20, c="black")
plt.scatter(new[0], new[1], marker="^")
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker="D")

plt.xlabel("X")
plt.ylabel("Y")
plt.show()
