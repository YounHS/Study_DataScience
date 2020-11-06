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

5. apply와 매개변수 lambda가 무엇인지? [#13](https://github.com/YounHS/Study_DataScience/issues/13#issue-736162649)

pandas의 일반적인 함수는 DataFrame내 함수를 사용하면 되지만, 커스텀 함수를 DataFrame에 적용하려면 map(), apply(). applymap()을 사용해야한다.



- map()

  - DataFrame 타입이 아니라, 반드시 Series 타입에서만 사용해야함

    ```
    값(value) + 인덱스(index) = 시리즈 클래스(series)
    ```

  - Series는 Numpy에서 제공하는 1차원 배열과 유사하며 각 데이터의 의미를 표시하는 index 추가 가능

  - 데이터 자체는 그냥 값의 1차원 배열

  - Series 값을 하나씩 꺼내서 lambda 함수의 인자로 넘기는 커스텀 함수를 각 value 별로 실행

  - example

    ```python
    import pandas as pd
    
    data = {'team ' : ['russia', 'saudiarabia', 'egypt', 'uruguay']
            'against': ['saudiarabia', 'russia', 'uruguay', 'egypt'],
            'fifa_rank': [65, 63, 31, 21]}
    columns = ['team', 'against', 'fifa_rank']
    
    # index는 0, 1, 2, 3
    df = pd.DataFrame(data, columns = columns)
    
    # 팀의 통산성적을 반환하는 커스텀 함수
    def total_record(team):
        ...
        # calculation from Database
        ...
        return win_count, draw_count, lose_count, winning_rate
    
    # team 컬럼에 있는 데이터들에 대해 각각 통산성적에 대한 winning_rate를 구한 다음 DataFrame에 추가하고자 한다면 하단과 같이 작성한다. 선택한 컬럼 'team'은 Series 객체이므로 map 함수를 사용한다.
    df['winning_rate'] = df['team'].map(lambda x: total_record(x)[3])
    ```



- apply()
  - 커스텀 함수를 사용하기 위해 DataFrame에서 복수 개의 컬럼이 필요하다면 apply 함수를 사용해야함

  - example(map()에서 사용한 example 이어감)

    ```python
    # 팀과 상대팀 간의 상대전적을 반환하는 커스텀 함수
    def relative_record(team, against):
        ...
        # calculation from Database
        ...
        return win_count, draw_count, lose_count, winning_rate
    
    # team과 against 컬럼의 값들을 각각 인자로 넘겨, 각 팀의 상대팀 대상 상대전적에 대한 winning_rate를 구한 다음 DataFrame에 추가하고자 한다면 lambda 함수와 커스텀 함수로 넘길 인자가 2개이므로, DataFrame의 apply 함수를 사용하되, 함수를 적용할 대상이 각각의 로우에 해당하므로 axis를 1로 지정하여 넘긴다.
    df['winning_rate'] = df.apply(lambda x: relative_record(x['team'], x['against'])[3], axis=1)
    ```



- applymap()
  - DataFrame클래스의 함수이긴 하나, apply()처럼 각 row(axis=1)나 각 column(axis=0)별로 작동하는 함수가 아니라, 각 요소별로 작동
  - 인자로 전달하는 커스텀함수가 Single value로부터 Single value를 반환한다는 점이 중요



- index 값을 apply 함수에 적용하는 방법
  - apply() 자체가 각각의 row를 꺼내올 때, row를 Series 객체로써 커스텀 함수를 적용하는 것이 아닌 numpy 객체로써 커스텀 함수를 적용하기 때문

    ```python
    ~.apply(lambda x: func(x.index))  # Error!
    
    ~.index.map(lambda x: func(x))  # run!
    ~.apply(lambda x: func(x.name), axis=1)  # run!
    ```

------

6. select_dtypes(), to_frame(), join(), columns가 무엇인지? [#14](https://github.com/YounHS/Study_DataScience/issues/14#issue-736163829)

- select_dtypes()

  - object형 데이터와 비object형(숫자형) 데이터를 구분하여 호출하는 함수

    ```python
    # object형 데이터 값만 호출
    data.select_dtypes(include = 'object')
    
    # object형이 아닌 데이터 값만 호출
    data.select_dtypes(exclude = 'object')
    ```

    

- to_frame()

  - Series 형태의 데이터를 DataFrame으로 변환

  - example

    ```python
    s = pd.Seriese(['a', 'b', 'c'], name='vals')
    s.to_frame()
    ```

    

- join()

  - 리스트 형태를 문자열로 변환

  - example

    ```python
    time_str = '10:34:17'
    time_str.split(':')  # ['10', '34', '17']
    
    ':'.join(time_str)  # '10:34:17'
    ```

    

- columns
  
  - 입력된 columns는 열의 레이블의 리스트를 입력 받음

------

7. index.values가 무엇인지? [#15](https://github.com/YounHS/Study_DataScience/issues/15#issue-736165595)

- index.values

  - index 객체를 실제 값 array로 변환

  - (판다스 공식 사이트) 기본 데이터에 대한 참조가 필요한지 아니면 NumPy 어레이에 대한 참조가 필요한지에 따라 'Index.array' 또는 'Index.to_numpy()'를 사용하는 것을 추천

  - example

    ```python
    #Index 객체를 실제 값 array로 변환 
    df.index.values
    ```

------

8. drop, remove의 차이가 무엇인지? [#16](https://github.com/YounHS/Study_DataScience/issues/16#issue-736166238)

- drop

  - pandas의 Series나 DataFrame 객체에서 row나 column을 삭제하기 위해 사용

  - example

    ```python
    # Series
    obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
    new_obj = obj.drop('c')
    
    # DataFrame (columns 삭제도 가능)
    data = DataFrame(np.arange(16).reshape((4, 4))),
    				index=['Ohio', 'Colorado', 'Utah', 'New York'],
        			columns=['one', 'two', 'three', 'four']
    data.drop(['Colorado', 'Ohio'])
    data.drop(['two', 'four'], axis=1)
    ```

    

- remove

  - python의 list에서 원소를 삭제하기 위해 사용

  - example

    ```python
    # del -> '2' 값이 삭제
    a = [1, 2, 3, 4, 5, 6, 7]
    del a[1]
    
    # remove -> '3' 값이 삭제
    a = [1, 2, 3, 4, 5, 6, 7]
    a.remove(3)
    ```

------

9. min(), .min() 매개변수 사용법과 np.log1p()에서 1p가 무엇인지? [#17](https://github.com/YounHS/Study_DataScience/issues/17#issue-736167135)

- python의 min()
  - 인수로 받은 자료형 내에서 최소값을 찾아서 반환하는 함수
  - 인수는 iterable(반복 가능한) 자료형 사용
  - min(iterable)
    - 매개변수로 들어온 인자 내부에서 제일 작은 값 반환
  - min(arg1, arg2)
    - 매개변수로 들어온 iterable 인자들 중 가장 작은 인자를 반환



- pandas의 min()

  - 그룹화되는 데이터의 최소값

  - example

    ```python
    df.groupby('data').min()
    ```

    

- np.log1p()의 1p
  - numpy에 0이 포함된 배열을 np.log()에 대입하면 **RuntimeWarning: divide by zero encountered in log** 라는 경고메시지 출력
  - np.log1p()를 사용하여 x+1을 수행

------

10. KNNImputer(n_neighbors), transform()가 무엇인지? [#18](https://github.com/YounHS/Study_DataScience/issues/18#issue-736167745)

- KNNImputer(n_neighbors)
  - 결측값을 대치하는 데 널리 사용되는 방법
  - KNN(K-최근접-이웃 알고리즘) 특성을 사용
  - n_neighbor 매개변수의 가장 가까운 이웃에 균일한 가중치를 부여



- transform()
  - fit()은 입력한 데이터에 특정 알고리즘 또는 전처리를 적용하는 함수이며 이를 통해 transformer에 알맞는 파라미터를 생성할 수 있음
  - fit()을 통해 생성된 파라미터를 통해서 모델을 적용시켜 데이터셋을 알맞게 변환하는 함수
  - fit_transform()은 같은 데이터셋을 사용하여 fit과 transform을 한 번에 하는 함수

------

11. concat(objs)에서 objs 매개변수가 무엇인지? [#19](https://github.com/YounHS/Study_DataScience/issues/19#issue-736168175)

- concat()

  - example and parameter mean

    ```python
    pd.concat(objs=[a, b],  # Series, DataFrame, Panel object 
              axis=0,  # 0: 위+아래로 합치기, 1: 왼쪽+오른쪽으로 합치기 
              join='outer', # 'outer': 합집합(union), 'inner': 교집합(intersection) 
              join_axes=None, # axis=1일 경우 특정 DataFrame의 index를 그대로 이용하려면 입력 
              ignore_index=False,  # False: 기존 index 유지, True: 기존 index 무시 
              keys=None, # 계층적 index 사용하려면 keys 튜플 입력 
              levels=None, 
              names=None, # index의 이름 부여하려면 names 튜플 입력 
              verify_integrity=False, # True: index 중복 확인 
              copy=True) # 복사
    ```

------

12. KFold(), cross_val_score가 무엇인지? [#20](https://github.com/YounHS/Study_DataScience/issues/20#issue-737009201)

- KFold()

  - train_test_split()의 반복

  - train_test_split()

    - 데이터의 분리 비 (ex. 4:6)

    - example

      ```python
      from sklearn.model_selection import train_test_split
      train_test_split(arrays, test_size, train_size, random_state, shuffle, stratify)
      
      # Parameter
      # arrays : 분할시킬 데이터를 입력 (Python list, Numpy array, Pandas dataframe 등..)
      # test_size : 테스트 데이터셋의 비율(float)이나 갯수(int) (default = 0.25)
      # train_size : 학습 데이터셋의 비율(float)이나 갯수(int) (default = test_size의 나머지)
      # random_state : 데이터 분할시 셔플이 이루어지는데 이를 위한 시드값 (int나 RandomState로 입력)
      # shuffle : 셔플여부설정 (default = True)
      # stratify : 지정한 Data의 비율을 유지한다. 예를 들어, Label Set인 Y가 25%의 0과 75%의 1로 이루어진 Binary Set일 때, stratify=Y로 설정하면 나누어진 데이터셋들도 0과 1을 각각 25%, 75%로 유지한 채 분할된다.
      
      # Return
      # X_train, X_test, Y_train, Y_test : arrays에 데이터와 레이블을 둘 다 넣었을 경우의 반환이며, 데이터와 레이블의 순서쌍은 유지된다.
      # X_train, X_test : arrays에 레이블 없이 데이터만 넣었을 경우의 반환
      ```

      ```python
      import numpy as np
      from sklearn.model_selection import train_test_split
      
      X = [[0,1],[2,3],[4,5],[6,7],[8,9]]
      Y = [0,1,2,3,4]
      
      # 데이터(X)만 넣었을 경우
      X_train, X_test = train_test_split(X, test_size=0.2, random_state=123)
      # X_train : [[0,1],[6,7],[8,9],[2,3]]
      # X_test : [[4,5]]
      
      # 데이터(X)와 레이블(Y)을 넣었을 경우
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=321)
      # X_train : [[4,5],[0,1],[6,7]]
      # Y_train : [2,0,3]
      # X_test : [[2,3],[8,9]]
      # Y_test : [1,4]
      ```

      

  - 데이터의 수가 적은 경우 신뢰도가 떨어지며, 그렇다고 검증 데이터의 수를 증가시키면 학습 데이터의 수가 적어지는 딜레마를 해결하기 위한 검증방법

  - example

    ```python
    from sklearn.model_selection import KFold
    
    cv = KFold(n_splits=3,random_state=1,shuffle=False)
            # n_splits=3이면  [훈련,검증] 3개만들어준다. 
    
    실제 적용
    
    for t,v in cv.split(train):
        train_cv=train.iloc[t]       # 훈련용
        val_cv=train.iloc[v]         # 검증용 분리.
        
        train_X=train_cv.loc[:,'독립변수들']    # 훈련용 독립변수들의 데이터,
        train_Y=train_Cv.loc[:,'종속변수만']    # 훈련용 종속변수만 있는 데이터
         
        val_X=val_cv.loc[:,'독립변수들']        # 검증용 독립변수들의 데이터,
        val_Y=val_Cv.loc[:,'종속변수만']        # 검증용 종속변수만 있는 데이터,
    ```

    

- cross_val_score

  - 단순 교차 검증

  - 파라미터는 (모델명, 훈련데이터, 타겟, cv)

  - cv는 폴드(Fold) 수를 의미 (default = 3)

  - example

    ```python
    from sklearn.model_selection import cross_val_score
    
    logreg = LogisicRegression()
    score = cross_val_score(logreg, train, test, cv=5)
    ```

------

13. residual, RidgeCV()가 무엇인지? [#21](https://github.com/YounHS/Study_DataScience/issues/21#issue-737009841)

- residual (잔차)
  - 이상치 데이터 포인트 확인
  - 피팅 모델 적합도 평가
  - 오차 분산의 상수 여부 확인
  - 오차항들의 정규분포 여부 평가
  - 절대값이 큰 잔차는 회귀 모형에 문제가 있다는 것을 의미
  - 잔차 plot이 깔대기 모양을 보여준다면(경향성을 보인다면) 잔차의 크기는 원인변수 x에 종속되어 있음을 의미



- RidgeCV() -> ***파라미터 공부 더 필요***

  - 교차 검증 방법

  - example

    ```python
    from sklearn.linear_model import RidgeCV
    
    RCV = RidgeCV(alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, normalize=False, scoring=None, cv=None, gcv_mode=None, store_cv_values=False)
    
    # Parameter
    # alphas : ndarray of shape (n_alphas,), default=(0.1, 1.0, 10.0)
    # alphas : 정규화 강도. 양의 부동소수점
    # fit_intercept : bool, default=True
    # fit_intercept : 이 모형에 대한 절편을 계산할지 여부. false -> 절편이 계산에 사용 X
    # normalize : bool, default=False
    # normalize : fit_intercept가 False로 설정된 경우 이 파라미터는 무시. True이면 회귀 분석 전에 평균을 뺀 후 l2-norm로 나누면 역률 X가 정규화됩니다.
    # scoring : string, callable, default=None
    # cv : int, cross-validation generator or an iterable, default=None
    # gcv_mode : {‘auto’, ‘svd’, eigen’}, default=’auto’
    # store_cv_values : bool, default=False
    ```

------

14. np.expm1()가 무엇인지? [#24](https://github.com/YounHS/Study_DataScience/issues/24#issue-737012447)

- expm1()

  - numpy 내장 함수

  - 로그 함수 log1p()로 변환된 값을 원래 값으로 변환

  - 입력 어레이 값에 대해 exp(x) -1을 계산

  - np.exp() - 1 보다 더 높은 정확도를 제공

  - example

    ```python
    import numpy as np
    
    a = np.expm1(1e-10)
    b = np.exp(1e-10) - 1
    
    print('np.expm1:', a)
    print('np.exp:', b)
    # np.expm1: 1.00000000005e-10
    # np.exp: 1.000000082740371e-10
    ```

------

