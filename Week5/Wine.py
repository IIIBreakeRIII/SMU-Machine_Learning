import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cross_Validate : 교차 검증, 분리한 데이터를 교차하여 모델을 검증
# GridSearchCV : 분류 or 회귀 알고리즘에 사용되는 하이퍼파라미터를 순차적으로 입력해 학습을 하고 측정을 하면서 가장 좋은 파라미터를 알려줌
# RandomizedSearchCV : 모든 조합을 다 시도하지는 않고, 각 반복마다 임의의 값만 대입해 지정한 횟수마큼 평가/Users/breaker/Desktop/BreakeR/SMU/ML/Week3/Code/FuelAndElectric.py 
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV

# StandardScaler : 표준화
from sklearn.preprocessing import StandardScaler

# LogisticRegression : 로지스틱 회귀
from sklearn.linear_model import LogisticRegression

# DecisionTreeClassifier : 결정 트리, 데이터에 있는 규칙을 학습을 통해 자동으로 찾아내 트리 기반의 분류 규칙을 만드는 알고리즘
# plot_tree : Decision Tree 시각화
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Scipy : 과학적 컴퓨팅 영역의 여러 기본적인 작업을 위한 라이브러리
# uniform : random 모듈 안에 정의되어 있는 두 수 사이의 랜덤한 소수를 리턴해주는 함수
# randint : 두 정수 사이의 랜덤한 정수를 리턴시켜주는 함수
from scipy.stats import uniform, randint

# 파일 읽어오기
wine = pd.read_csv('wine.csv')

# numpy 배열로 바꾸기
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 데이터 분류
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# StandardScaler 선언
# train_input 학습
ss = StandardScaler()
ss.fit(train_input)

# 정규화
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 로지스틱 회귀
lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.coef_, lr.intercept_)
