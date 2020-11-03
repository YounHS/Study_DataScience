# Questions
1. np.log 무엇인지? [#9](https://github.com/YounHS/Study_DataScience/issues/9#issue-734545011)

- numpy.log 함수

- 입력 Array의 자연로그 값을 반환하며, 밑이 10 또는 2인 로그는 log10, log2를 사용함.

- example

  ```python
  import numpy as np
  
  a = np.array([1, np.e, np.e**2, 0])
  print(np.log(a))
  ```

- 문제가 됬던 부분의 코드

  ```python
  df_train['SalePrice'] = np.log(df_train['SalePrice'])
  ```

---

2. StandardScaler가 무엇인지? [#10](https://github.com/YounHS/Study_DataScience/issues/10#issue-735030966)

- Scikit-Learn의 스케일러 기법 중 한가지
- Scikit-Learn의 대표적인 스케일러 기법

|      | 종류           | 설명                                                         |
| ---- | -------------- | ------------------------------------------------------------ |
| 1    | StandardScaler | 기본 스케일. 평균과 표준편차 사용                            |
| 2    | MinMaxScaler   | 최대/최소값이 각각 1, 0이 되도록 스케일링                    |
| 3    | MaxAbsScaler   | 최대절대값과 0이 각각 1, 0이 되도록 스케일링                 |
| 4    | RobustScaler   | 중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화 |



- StandardScaler
  - 평균을 제거하고 데이터를 단위 분산으로 조정
  - 이상치가 있다면 평균과 표준편차에 영향으 ㄹ미쳐 변환된 데이터의 확산은 매우 달라짐
  - 이상치가 있는 경우, 균형 잡힌 척도 보장 불가
  - example

```python
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
print(standardScaler.fit(train_data))
train_data_standardScaled = standardScaler.transform(train_data)
```



- MinMaxScaler
  - 모든 feature 값이 0 ~ 1 사이에 있도록 데이저 재조정
  - 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축될 가능성 존재
  - StandardScaler와 마찬가지로, 이상치에 민감
  - example

```python
from sklearn.preprocessing import MinMaxScaler
minMaxScaler = MinMaxScaler()
print(minMaxScaler.fit(train_data))
train_data_minMaxScaled = minMaxScaler.transform(train_data)
```



- MaxAbsScaler
  - 절대값이 0 ~ 1 사이에 매핑되도록 설정 (-1 ~ 1 사이로 재조정)
  - 양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작
  - 큰 이상치에 민감할 가능성 존재
  - example

```python
from sklearn.preprocessing import MaxAbsScaler
maxAbsScaler = MaxAbsScaler()
print(maxAbsScaler.fit(train_data))
train_data_maxAbsScaled = maxAbsScaler.transform(train_data)
```



- RobustScaler
  - 이상치의 영향을 최소화한 기법
  - median 값과 IQR을 사용하기 때문에 StandardScaler와 비교해보면 표준화 후 동일한 값을 더 넓게 분포 시키는 것을 확인 가능
  - IQR = Q3 - Q1 : 즉, 25퍼센타일과 75퍼센타일의 값들을 다룸
  - example

```python
from sklearn.preprocessing import RobustScaler
robustScaler = RobustScaler()
print(robustScaler.fit(train_data))
train_data_robustScaled = robustScaler.transform(train_data)
```

​																																		[참고 사이트로 이동](https://mkjjo.github.io/python/2019/01/10/scaler.html)

------

3. scipy의 stats가 무엇인지? [#10](https://github.com/YounHS/Study_DataScience/issues/11#issue-735031862)

- scipy는 각종 수치 해석 기능을 제공하는 파이썬 패키지
- scipy.stats 서브패키지는 여러가지 확률 분포 분석을 위한 기능을 제공

| 종류 | 이름                | 확률 분포                 |
| :--: | ------------------- | ------------------------- |
| 이산 | bernoulli           | 베르누이 분포             |
| 이산 | binom               | 이항 분포                 |
| 연속 | uniform             | 균일 분포                 |
| 연속 | norm                | 가우시안 정규 분포        |
| 연속 | beta                | 베타 분포                 |
| 연속 | gamma               | 감마 분포                 |
| 연속 | t                   | 스튜던트 t 분포           |
| 연속 | chi2                | 카이 제곱 분포            |
| 연속 | f                   | F 분포                    |
| 연속 | dirichlet           | 디리클리 분포             |
| 연속 | multivariate_normal | 다변수 가우시안 정규 분포 |

​																																		[참고 사이트로 이동](https://namyoungkim.github.io/scipy/probability/2017/09/04/scipy/)

- 이외에 왜도, 첨도를 그래프 출력 가능

  ```python
  from scipy.stats import skew, kurtosis
  
  # 왜도
  skew(data)
  
  # 첨도
  kurtosis(data, fisher=True)
  ```

------

4. scipy의 stats가 무엇인지? [#11](https://github.com/YounHS/Study_DataScience/issues/11#issue-735031862)

- 왜도 (Skewness)
  - 분포의 비대칭도
  - 정규분포 -> 왜도 = 0
  - 왼쪽으로 치우짐 -> 왜도 > 0
  - 오른쪽으로 치우침 -> 왜도 < 0

- 첨도 (Kurtosis)
  - 확률분포의 뾰족한 정도
  - 정규분포 -> 첨도 = 0 (Pearson 첨도 = 3)
  - 위로 뾰족함 -> 첨도 > 0 (Pearson 첨도 > 3)
  - 아래로 뾰족함 -> 첨도 < 0 (Pearson 첨도 < 3)
  - **Fisher = True**
    - 첨도 기준이 Fisher (normal ==> 0.0)
    - 정규분포의 첨도 = 0
  - **Fisher = False**
    - 첨도 기준이 Pearson (normal ==> 3.0)
    - 정규분포의 첨도 = 3

------

