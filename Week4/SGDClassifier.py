import pandas as pd
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

sc = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

sc.partial_fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# Early Exit

sc = SGDClassifier(loss="log_loss", random_state=42)
train_score = []
test_score = []

classes = np.unique(train_target)

for _ in range(0, 300):
	sc.partial_fit(train_scaled, train_target, classes=classes)
	train_score.append(sc.score(train_scaled, train_target))
	test_score.append(sc.score(test_scaled, test_target))

sc = SGDClassifier(loss='log_loss', max_iter=10000, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
