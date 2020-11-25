# ML Algorithm
1. LogisticRegression (로지스틱 회귀)

> 목표: 
> **로지스틱 함수를 구성하는 계수와 절편에 대해 Log Loss를 최소화하는 값을 찾는 것**

- 회귀를 사용하여 데이터가 어떤 범주에 속할 확률을 0 ~ 1 사이의 값으로 예측

- 그 확률에 따라 가능성이 더 높은 범주에 속하는 것으로 분류해주는 지도 학습 알고리즘

- ex) 스팸 메일 분류기

- Log-Odds

  - 개념

    - 선형 회귀는 각 속성의 값에 계수를 곱하고 절편을 더하여 예측 값 산출
    - 로지스틱 회귀는 예측 값 대신 log-odds 산출
    - Odds = event occur/event not occur
    - Log-Odds = log(Odds)

  - 계산

    - dot product 방식으로 Log-Odds 산출

    - 각 속성들의 값이  포함된 행렬, 그 속성들 각각의 계수가 포함된 행렬을 하단과 같이 계산 가능

      <br><img src="https://github.com/YounHS/Study_DataScience/blob/master/3.%20ML/picture/LR_logodds.png" width="40%"><br>

    - 연산은 numpy의 `np.dot()`으로 쉽게 처리 가능

      ```python
      log_odds = np.dot(features, coefficients) + intercept
      ```

- Sigmoid Function

  - 확률을 0 ~ 1 사이의 커브 모양으로 나타내주는 것을 가능하게 해주는 함수
  - numpy에서 `np.exp(-z)`로 쉽게 계산 가능

- Log Loss (로그 손실)

  - 로지스틱 회귀가 확률을 제대로 예측하는지 확인해주는 함수

  - 모델의 적합성을 평가하기 위해 각 데이터 샘플의 손실을 계산 후, 그것들의 평균화 필요

  - 경사하강법(Gradient Descent)을 사용하여 모든 데이터에서 Log Loss를 최소화하는 계수 산출 가능

    <br><img src="https://github.com/YounHS/Study_DataScience/blob/master/3.%20ML/picture/LR_logloss.png" width="40%"><br>

    - m: 데이터 총 개수
    - y_i: 데이터 샘플 i의 분류
    - z_i: 데이터 샘플 i의 log-odd
    - h(z_i): 데이터 샘플 i의 log-odd의 sigmoid (데이터 샘플 i가 분류에 속할 확률)

- Classification Threshold (임계값)
  
  - 대부분 알고리즘에서 Default 임계 값은 0.5 (필요에 따라 변경 가능)

> **요약**
>
> - 로지스틱 회귀 분석은 데이터 샘플을 1 또는 0 클래스 둘 중 어디에 속하는지 이진 분류를 수행하여 예측
> - 각 features들의 계수 Log Odds를 구한 후, Sigmoid 함수를 적용하여 실제로 데이터가 해당 클래스에 속할 확률을 0 ~ 1 사이의 값으로 산출
> - Loss Function은 ML 모델이 얼마나 잘 예측했는지 확인하는 방법
> - 데이터가 클래스에 속할지 말지 결정할 확률 컷오프를 Threshold (임계값)이라고 함
> - Scikit-learn을 통해 모델 생성 및 각 feature들의 계수 산출이 가능하며, 이 때 각 계수(coefficient)들은 데이터를 분류함에 있어 해당 속성이 얼마나 중요한지 해석하는 데에 사용 가능

------

2. KNN (K-최근접이웃)

> 목표: 
> **새로운 데이터가 주어졌을 때 기존 데이터 가운데 가장 가까운 k개 이웃의 정보로 새로운 데이터 예측하는 것**

- 학습이라고 할만한 절차가 없음
- 모델을 별도로 구축하지 않는다는 뜻으로 게으른 모델(Lazy model) 또는 Instance-based Learning으로 불림
- 데이터로부터 모델을 생성해 과업을 수행하는 Model-based Learning과 대비
- 별도 모델 생성 과정없이 각각의 관측치(instance)만을 이용하여 분류/회귀 등 과업을 수행
- KNN의 하이퍼파라미터 (2가지)
  - 탐색할 이웃 수(k)
    - k가 작을 경우, 데이터의 지역적 특성을 지나치게 반영(overfitting)
    - k가 클 경우, 모델이 과하게 정규화되는 경향(underfitting)
  - 거리 측정 방법
    - Euclidean Distance
    - Manhattan Distance
    - Mahalanobis Distance
    - Correlation Distance
    - Rank Correlation Distance
- best K 찾기
  - 학습데이터와 검증데이터를 나누고, k값에 변화를 주면서 실험 필요
  - besk K 찾는 작업을 대신해주는 라이브러리 사용 필요
- combining rule
  - 이웃들 범주 가운데 빈도 기준 제일 많은 범주로 새 데이터의 범주를 예측하는 다수결 결정 방식
  - 거리가 가까운 이웃의 정보에 좀 더 가중치를 부여하는 가중합 방식
- cut-off 기준 설정
  - 학습데이터 범주의 사전확률을 고려해야함
  - ex) 제조업 정상/불량 데이터 분류의 경우, 0.7:0.3 보단 0.8:0.2가 합리적
- KNN의 장점
  - 학습데이터 내에 끼어있는 노이즈의 영향을 크게 받지 않음
  - 학습데이터 수가 많다면 꽤 효과적인 알고리즘
  - Mahalanobis 처럼 데이터의 분산을 고려할 경우, 매우 강건한 방법론
  - 1-NN에 한해, 모델 성능을 어느 정도 보장 가능
- KNN의 단점
  - 최적 이웃의 수(k)와 어떤 거리척도가 분석에 적합한지 불분명
  - 상기로 인해 데이터 각각의 특성에 맞게 연구자가 임의로 선정해야함
  - 새로운 관측치와 각각의 학습 데이터 사이의 거리를 전부 측정해야하므로 오래 걸리는 계산 시간

> **참고**
>
> - KNN 수행 전 반드시 변수를 정규화해야함
> - 명목/범주형 데이터의 경우, one-hot encoding을 사용해 더미 변수로 만들어줘야함
> - KNN의 계산복잡성을 줄이려는 Locality Sensitive Hashing, Network based Indexer, Optimized product quantization 등이 제안됨

------

3. RandomForest (랜덤 포레스트)

> 목표: 
> **여러 개의 결정 트리 분류기가 배깅을 기반으로 각자의 데이터를 샘플링하여 학습 후, 최종 보팅을 통해 예측하는 것**

- **Bagging(배깅)**

  - Bootstrap Aggregating의 약자
  - Voting과 달리 동일한 알고리즘으로 여러 분류기를 만든 후, 예측 결과를 보팅으로 최종 결정하는 알고리즘
  - 배깅 진행 과정
    1. 동일한 알고리즘을 사용하는 일정 수의 분류기 생성
    2. 각 분류기는 bootstraping 방식으로 생성된 샘플 데이터를 학습
    3. 최종적으로 모든 분류기의 보팅을 통해 예측 결정

- 여러 개의 결정 트리를 활용한 배깅 방식의 대표적인 알고리즘

- 장점

  - 결정 트리의 쉽고 직관적인 장점을 그대로 지님
  - 앙상블 알고리즘 중 비교적 빠른 수행속도
  - 다양한 분야에서 좋은 성능 나타냄

- 단점

  - 하이퍼 파라미터가 많아 튜닝을 위해 많은 시간 소요

- 하이퍼파라미터

  - n_estimators

    - 사용되는 결정 트리의 개수 지정
    - default: 10
    - 무작정 트리 개수를 늘리면 성능 향상 대비 오랜 시간 소요

  - min_samples_split

    - 노드를 분할하기 위한 최소한의 샘플 데이터 수

      -> 과적합 제어에 사용. 값이 작을수록 분할 노드가 많아져 과적합 가능성 증가

    - default: 2

  - min_samples_leaf

    - 리프노드가 되기 위한 최소한의 샘플 데이터 수

      -> 과적합 제어에 사용. 값이 작을수록 과적합 가능성 증가

    - default: 1

    - 불균형 데이터의 경우, 특정 클래스 데이터가 극도로 적을 수 있으므로 작은 값 설정 필요

  - max_features

    - 최적의 분할을 위해 고려할 피처의 치대 개수

    - default: auto

      -> 결정 트리에서는 default가 Non인 것과 차이

    - int형으로 지정 (피처 개수)

    - float형으로 지정 (전체 개수의 일정 비율만큼 사용)

    - `sqrt` 또는 `auto` (전체 피처 중 √(피처 개수)만큼 선정)

    - `log2` (전체 피처 중 log2(전체 피처 개수)만큼 선정)

  - max_depth

    - 트리의 최개 깊이

    - default: None

      -> 완벽하게 클래스 값이 결정될 때까지 분할

      -> 또는 데이터 개수가 min_samples_split보다 작아질 때까지 분할

    - 깊이가 깊어지면 과적합될 수 있으므로 적절한 제어 필요

  - max_leaf_nodes

    - 리프노드의 최대 개수
    - default: None

> **참고**
>
> - from sklearn.model_selection import GridSearchCV 를 통해 best param 추출 가능
> - scikit-learn에서는 RandomForestClassifire 클래스를 통해 랜덤 포레스트 기반의 분류 지원

------

4. SVM (Support Vector Machine)

> 목표: 
> **주어진 데이터가 어느 그룹에 속하는지 분류**

- 분류 사이에 존재하는 여백을 의미하는 마진을 최대화하여 일반화 능력을 극대화하는 모델

  - 마진: SVM에서 데이터를 분류하기 위해 사용한 일종의 분류선과 가장 가까운 데이터들과의 거리
  - 분류선을 Decision Boundary라 부르고 이 선과 가장 가까운 데이터를 Support Vector라 명명

- 결정 경계

  - Boundary로부터 Vector가 멀리 떨어질수록 최적의 결정 경계
  - 데이터에 2개의 속성이 있다면 결정 경계는 간단한 선 형태
  - 데이터에 3개의 속성이 있다면 결정 경계는 평면 형태
    - 차원이 늘어나 고차원이 되면 나타나는 결정 경계는 "초평면" 이라 명명

- 장점

  - Decision Boundary를 정의하는 것이 Support Vector이므로 데이터 포인트 중 Support Vector만 잘 골라내면 나머지 쓸데 없는 데이터 포인트 무시 가능 (따라서 속도가 빠름)

- 허용 가능한 오류 범위 내에서 가능한 최대 마진을 만들려 함

- 파라미터 C

  - 허용되는 오류 양 조절
  - C 값이 클수록 오류를 덜 허용 (하드 마진)
  - C 값이 작을수록 오류를 더 허용 (소프트 마진)

- 선형으로 분리할 수 없는 점들을 분류하기 위해 커널 사용

  - 커널: 원래 가지고 있는 데이터를 더 높은 차원의 데이터로 변환

  - 2차원의 점으로 나타낼 수 있는 데이터

    -> 다항식 커널은 3차원으로 변환

    -> RBF  커널은 점을 무한한 차원으로 변환

    - RBF 커널의 파라미터 감마
      - 감마가 너무 크면 학습 데이터에 너무 의존하여 오버피팅 발생 가능


------

