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

