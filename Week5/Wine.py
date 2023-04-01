import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from scipy.stats import uniform, randint

# 트리 알고리즘
# DataFrame ==> np ==> 표준화(평균을 중심으로 동일한 표준편차)
wine = pd.read_csv('wine.csv')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.coef_, lr.intercept_)

dt = DecisionTreeClassifier(max_depth=3, random_state = 42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

print(dt.feature_importances_)

train_input, test_inpput, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

scores = cross_validate(dt, train_input, train_target)
print(scores)
print(np.mean(scores['test_score']))

params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target))

print(gs.best_params_)

print(gs.cv_results_['mean_test_score'])

rgen = randint(0, 10)
rgen.rvs(10)

np.unique(rgen.rvs(1000), return_counts=True)

ugen = uniform(0, 1)
ugen.rvs(10)

params = {'min_impurity_decrease': uniform(0.0001, 0.001), 'max_depth': randint(20, 50), 'min_samples_split': randint(2, 25), 'min_samples_leaf': randint(1, 25)}

gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)

print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))

dt = gs.best_estimator_
print(dt.score(test_input, test_target))
