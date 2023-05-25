# 데이터 로드
import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt

# 삼성전자
df = fdr.DataReader('005930', '2018-05-04', '2022-12-31')

print(df.shape)
print(df.keys())
