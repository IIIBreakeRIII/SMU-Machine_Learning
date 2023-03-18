import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

LoadInput = np.load("Week2/HomeWork/data_input.npy")
LoadTarget = np.load("Week2/HomeWork/data_target.npy")

train_input, test_input, train_target, test_target = train_test_split(LoadInput, LoadTarget, stratify=LoadTarget, random_state = 1800)

kn = KNeighborsClassifier()

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std

kn.fit(train_scaled, train_target)
print(kn.score(test_scaled, test_target))

# predict classes for all points in X-Y plane
xx, yy = np.meshgrid(np.linspace(-2, 2, 500), np.linspace(-2, 2, 500))
Xfull = np.column_stack((xx.ravel(), yy.ravel(), np.ones_like(xx.ravel()), np.ones_like(xx.ravel())))
y_pred = kn.predict(Xfull)
y_pred = y_pred.reshape(xx.shape)

# plot the decision boundary
plt.contourf(xx, yy, y_pred, alpha=0.3)

# plot the training points
plt.scatter(train_scaled[:, 0], train_scaled[:, 1], c=train_target)

plt.xlabel("X")
plt.ylabel("Y")
plt.show()
