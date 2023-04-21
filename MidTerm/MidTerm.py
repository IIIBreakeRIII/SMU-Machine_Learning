import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax

data = pd.read_csv("MidTerm/midterm.csv")
data.head()

input_data = data[['Weight', 'Length', 'Height', 'Width']].to_numpy()
input_target = data['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(input_data, input_target, random_state=42)

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

indexes = (train_target == 'Bream') | (train_target == 'Roach') | (train_target == 'Whitefish') | (train_target == 'Parkki') | (train_target == 'Perch') | (train_target == 'Smelt')
train_indexes = train_scaled[indexes]
test_indexes = train_target[indexes]

lr = LogisticRegression()
lr.fit(train_indexes, test_indexes)

print(lr.predict(train_indexes[:4]))
print(np.round((lr.predict_proba(train_indexes[:4])), decimals=3))

decisions = lr.decision_function(train_indexes[:4])

lr.decision_function(test_scaled[:4])
proba = softmax(decisions)