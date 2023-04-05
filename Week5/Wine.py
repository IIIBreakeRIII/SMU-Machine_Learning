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
# 분류 모델
# 이진 분류 : True / False 로 나타남
# 다중 분류 : 종속형 변수가 2개 이상의 카테고리로 분류됨
# 위키피디아 : 독립 변수의 선형 결합을 이용하여 사건의 발생 가능성을 예측하는 데
# 사용되는 통계 기법
lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.coef_, lr.intercept_)

# DecisionTreeClassifier 정의
# max_depth : Tree 최고 깊이(루트 노드를 제외한 노드의)
dt = DecisionTreeClassifier(max_depth = 3, random_state = 42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

# matplotlib 으로 트리 표현
plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# feature_importances_ : 변수 중요도(특성 중요도)
# 각 변수가 모델에서 예측하는 결과에 얼마나 큰 영향을 미치는지 나타내는 지표
print(dt.feature_importances_)

# 다른 모델들의 사용에 따른 비교 검증을 위한 테스트 세트 남겨두기
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

# 교차 검증
# DecisionTreeClassifier 모델을 train_input 값과 train_target값에 대한 교차 검증 수행
# 교차 검증의 경우 데이터를 여러 개의 폴드(fold)로 나누고
# 각 폴드를 훈련 및 평가에 사용하는 과정을 반복하여 모델의 일반화 성능을 평가
scores = cross_validate(dt, train_input, train_target)
print(scores)
print(np.mean(scores['test_score']))
