# ML Algorithm
1. LogisticRegression (로지스틱 회귀)

> 목표: 
> **로지스틱 함수를 구성하는 계수와 절편에 대해 Log Log를 최소화하는 값을 찾는 것**

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