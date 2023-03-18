import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

LoadInput = np.load("data_input.npy")
LoadTarget = np.load("data_target.npy")

train_input, test_input, train_target, test_target = train_test_split(LoadInput, LoadTarget, stratify=LoadTarget, random_state = 1800)

# print(train_input)
# print(test_input)
# print(train_target)
# print(test_target)

kn = KNeighborsClassifier()

print("-------")
print(len(LoadInput))
print("-------")
print(len(LoadTarget))
print("-------")

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std

kn.fit(train_scaled, train_target)
print(kn.score(test_scaled, test_target))

# print("new1 =", kn.predict([new1]))
# print("new2 =", kn.predict([new2]))
# print("new3 =", kn.predict([new3]))
# print("new4 =", kn.predict([new4]))

# distances1, indexes1 = kn.kneighbors([new1])
# distances2, indexes2 = kn.kneighbors([new2])
# distances3, indexes3 = kn.kneighbors([new3])
# distances4, indexes4 = kn.kneighbors([new4])

plt.scatter(train_scaled[train_target==1, 0], train_scaled[train_target==1, 1], c='black')
plt.scatter(train_scaled[train_target==2, 0], train_scaled[train_target==2, 1], c="blue")
plt.scatter(train_scaled[train_target==3, 0], train_scaled[train_target==3, 1], c="red")
plt.scatter(train_scaled[train_target==4, 0], train_scaled[train_target==4, 1], c="green")

# for label in np.unique(train_target):
#     plt.scatter(train_scaled[train_target==label, 0], train_scaled[train_target==label, 1])

# plt.scatter(new1[0], new1[1], marker="^")
# plt.scatter(new2[0], new2[1], marker="^")
# plt.scatter(new3[0], new3[1], marker="^")
# plt.scatter(new4[0], new4[1], marker="^")
# plt.scatter(train_scaled[indexes1, 0], train_scaled[indexes1, 1], marker="D")
# plt.scatter(train_scaled[indexes2, 0], train_scaled[indexes2, 1], marker="D")
# plt.scatter(train_scaled[indexes3, 0], train_scaled[indexes3, 1], marker="D")
# plt.scatter(train_scaled[indexes4, 0], train_scaled[indexes4, 1], marker="D")

plt.xlabel("X")
plt.ylabel("Y")
plt.show()
