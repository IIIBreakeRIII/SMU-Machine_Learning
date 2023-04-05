import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cross_Validate : 교차 검증, 분리한 데이터를 교차하여 모델을 검증
# GridSearchCV : 분류 or 회귀 알고리즘에 사용되는 하이퍼파라미터를 
# 순차적으로 입력해 학습을 하고 측정을 하면서 가장 좋은 파라미터를 알려줌
# 하이퍼파라미터 : 모델의 성능을 조정하고 모델의 동작을 제어하는 매개 변수
# RandomizedSearchCV : 모든 조합을 다 시도하지는 않고, 각 반복마다 임의의 값만 대입해 지정한 횟수마큼 평가
# KFold : 학습 세트와 검증 세트를 나눠서 반복 검증하는 방식
# K 값만큼의 폴드 세트에 K번의 학습과 검증을 통해 K번 평가
# 과대 적합의 오류가 생길 수 있음
# StratifiedKFold : target에 속성값의 개수를 동일하게 하게 가져감으로서 KFold와 같이
# 데이터가 한 곳에 몰리는 것을 방지
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV, StratifiedKFold

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

print("lr.score(train_scaled, train_target) :", lr.score(train_scaled, train_target))
print("lr.score(test_scaled, test_target) :", lr.score(test_scaled, test_target))
print("lr.coef_, lr.intercept :", lr.coef_, lr.intercept_)
print("")

# DecisionTreeClassifier 정의
# max_depth : Tree 최고 깊이(루트 노드를 제외한 노드의)
dt = DecisionTreeClassifier(max_depth = 3, random_state = 42)
dt.fit(train_scaled, train_target)

print("dt.score(train_scaled, train_target) :", dt.score(train_scaled, train_target))
print("dt.score(test_scaled, test_target) :", dt.score(test_scaled, test_target))
print("")

# matplotlib 으로 트리 표현
plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# feature_importances_ : 변수 중요도(특성 중요도)
# 각 변수가 모델에서 예측하는 결과에 얼마나 큰 영향을 미치는지 나타내는 지표
print("dt.feature_importances_ :", dt.feature_importances_)
print("")

# 다른 모델들의 사용에 따른 비교 검증을 위한 테스트 세트 남겨두기
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

# 교차 검증
# DecisionTreeClassifier 모델을 train_input 값과 train_target값에 대한 교차 검증 수행
# 교차 검증의 경우 데이터를 여러 개의 폴드(fold)로 나누고
# 각 폴드를 훈련 및 평가에 사용하는 과정을 반복하여 모델의 일반화 성능을 평가
scores = cross_validate(dt, train_input, train_target)
print("scores :", scores)
print("np.mean(scores['test_score']) :", np.mean(scores['test_score']))
print("")

# StratifiedKFold(Default Parameters)
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print("np.mean(scores['test_score'])(After StratifiedKFold, Default Parameters) :", np.mean(scores['test_score']))
print("")

# StratifiedKFold
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print("np.mean(scores['test_score'])(After StratifiedKFold, Use Parameters) :", np.mean(scores['test_score']))
print("")

params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

# GridSearchCV는 머신러닝 알고리즘에 사용되는 하이퍼 파라미터를 입력해 
# 학습을 하고 측정을 하면서 가장 좋은 파라미터를 알려줌
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

dt = gs.best_estimator_
print('dt.score :', dt.score(train_input, train_target))
print('gs.best_params_ :', gs.best_params_)
print('gs.cv_results_ :', gs.cv_results_['mean_test_score'])
print("")

# 0부터 10사이의 랜덤한 정수를 반환해주는 변수 rgen
# rvs() = random variates의 약자로 지정된 분포에서 무작위 표본을 추출하는 함수
print("rgen.rvs(10)")
rgen = randint(0, 10)
rgen.rvs(10)
print("")

# rgen.rvs(1000) = rgen 객체에서 1000개의 무작위 표본을 추출하여 반환
# Example : rgen.normal(0, 1, size=1000) = 평균 0, 표준편차 1인 정규분포에서
# 1000개의 무작위 표본 추출
# return_counts의 경우, unique() 함수의 중복되지 않은 값의 배열과 함께
# 해당 값들이 출현한 빈도수를 반환하도록 설정
# 즉, unique() 함수가 중복되지 않은 값 배열과 그 값들의 출현 빈도수 배열 두개를 반환
# 아래 코드는, rgen 객체에서 생성된 1000개의 무작위 표본에서 각 값의 빈도수를 계산,
# 각 값과 빈도수를 포함하는 두 개의 배열을 반환
print("np.unique(ren.rvs(1000), return_counts=True)")
np.unique(rgen.rvs(1000), return_counts=True)
print("")

# 두 수 사이의 랜덤한 소수(0.001과 같은) 반환
print("ugen.rvs(10)")
ugen = uniform(0, 1)
ugen.rvs(10)
print("")

# 랜덤 서치
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
					"max_depth": randint(20, 50),
					"min_samples_split": randint(2, 25),
					"min_samples_leaf": randint(1, 25)}

gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)

print("gs.best_params_ :", gs.best_params_)

print("np.max(gs.cv_results_['mean_test_score']) :", np.max(gs.cv_results_['mean_test_score']))

dt = gs.best_estimator_
print("dt.score(test_input, test_target) :", dt.score(test_input, test_target))
