import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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

kn.fit(train_scaled, train_target)
print(kn.score(test_scaled, test_target))

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
