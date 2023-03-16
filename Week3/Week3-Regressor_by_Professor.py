import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

#%%
data_size=100
perch_length=np.random.randint(80,440,(1,data_size))/10 #(1,100)
perch_weight=perch_length**2-20*perch_length+110+np.random.randn(1,data_size)*50+100 #(1,100)

#결과 1
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
print(np.shape(perch_length))

train_input, test_input, train_target, test_target = train_test_split(
    perch_length.T, perch_weight.T, random_state=42)

print("학습데이터Shape: ", train_input.shape,"테스트데이터Shape: ", test_input.shape)

knr = KNeighborsRegressor()
# k-최근접 이웃 회귀 모델을 훈련합니다
knr.fit(train_input, train_target)
print("예측정확도",knr.score(test_input, test_target))

# 테스트 세트에 대한 예측을 만듭니다
test_prediction = knr.predict(test_input)
# 테스트 세트에 대한 평균 절댓값 오차를 계산합니다
mae = mean_absolute_error(test_target, test_prediction)
print("오류절대값:",mae)
