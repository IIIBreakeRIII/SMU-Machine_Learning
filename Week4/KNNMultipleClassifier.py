import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, softmax

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# KNN Multiple Classifier

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

print("------------------------------")
print("kn.classes_")
print("------------------------------")
print(kn.classes_)
print(" ")

print("------------------------------")
print("kn.predict(test_scaled[:5])")
print("------------------------------")
print(kn.predict(test_scaled[:5]))
print(" ")

proba = kn.predict_proba(test_scaled[:5])

print("------------------------------")
print("np.round(proba, decimals=4)")
print("------------------------------")
print(np.round(proba, decimals=4))
print(" ")

# Logistic Regression(Binary Classification)

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print("------------------------------")
print("lr.predict(train_bream_smelt[:5])")
print("------------------------------")
print(lr.predict(train_bream_smelt[:5]))
print(" ")

print("------------------------------")
print("lr.predict_proba(train_bream_smelt[:5])")
print("------------------------------")
print(lr.predict_proba(train_bream_smelt[:5]))
print(" ")

print("------------------------------")
print("lr.coef_, lr.intercept_")
print("------------------------------")
print(lr.coef_, lr.intercept_)
print(" ")

decisions = lr.decision_function(train_bream_smelt[:5])
print("------------------------------")
print("decisions")
print("------------------------------")
print(decisions)
print(" ")

print("------------------------------")
print("expit(decisions)")
print("------------------------------")
print(expit(decisions))
print(" ")

# Logistic Regression(Multiple Classification)

lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print("------------------------------")
print("lr.score(train_scaled, train_target) and lr.score(test_sclaed, test_target)")
print("------------------------------")
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
print(" ")

proba = lr.predict_proba(test_scaled[:5])
print("------------------------------")
print("np.round(proba, decimals=3)")
print("------------------------------")
print(np.round(proba, decimals=3))
print(" ")

print("------------------------------")
print("lr.coef_.shape, lr.intercept_.shape")
print("------------------------------")
print(lr.coef_.shape, lr.intercept_.shape)
print(" ")

# Soft Max Function

decision = lr.decision_function(test_scaled[:5])

print("------------------------------")
print("np.round(decision, decimals=2)")
print("------------------------------")
print(np.round(decision, decimals=2))
print(" ")

proba = softmax(decision, axis=1)
print("------------------------------")
print("np.round(proba, decimals=3)")
print("------------------------------")
print(np.round(proba, decimals=3))
print(" ")

