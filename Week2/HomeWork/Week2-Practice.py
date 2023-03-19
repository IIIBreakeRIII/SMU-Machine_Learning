import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

LoadInput = np.load("data_input.npy")
LoadTarget = np.load("data_target.npy")

train_input, test_input, train_target, test_target = train_test_split(LoadInput, LoadTarget, stratify=LoadTarget, random_state = 1800)

kn = KNeighborsClassifier(n_neighbors=5)
print("Neighbors : 5")

print("-------")
print("LoadInput :", len(LoadInput))
print("-------")
print("LoadTarget :", len(LoadTarget))
print("-------")

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std

kn.fit(train_scaled, train_target)
print("Predict Score :", kn.score(test_scaled, test_target))
print("--------")

test_data = ([20, 300] - mean) / std

distances, indexes = kn.kneighbors([test_data])
print("Test Data Predict =", kn.predict([test_data]))

plt.scatter(train_scaled[train_target==1, 0], train_scaled[train_target==1, 1], c='yellow')
plt.scatter(train_scaled[train_target==2, 0], train_scaled[train_target==2, 1], c="blue")
plt.scatter(train_scaled[train_target==3, 0], train_scaled[train_target==3, 1], c="red")
plt.scatter(train_scaled[train_target==4, 0], train_scaled[train_target==4, 1], c="green")
plt.scatter(test_data[0], test_data[1], marker="^", s=200)

plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker="X", c="black")

# for label in np.unique(train_target):
#     plt.scatter(train_scaled[train_target==label, 0], train_scaled[train_target==label, 1])

plt.xlabel("X")
plt.ylabel("Y")
plt.show()
