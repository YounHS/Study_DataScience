from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd

from sklearn.utils import shuffle
from processData import test_data, train_data, train_label, test

# 학습시키기 전, 주어진 데이터가 정렬되어 있어 학습에 방해가 될 수 있으므로 셔플링
train_data, train_label = shuffle(train_data, train_label, random_state=5)


# 모델 학습과 평가에 대한 파이프라인 생성
# scikit-learn에서 제공하는 fit(), predict()를 사용하면 매우 간단하게 학습과 예측 가능
def predictionModel(model):
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    accuracy = round(model.score(train_data, train_label) * 100, 2)
    print('Accuracy: ', accuracy, '%')
    return prediction


# Logistic Regression
log_pred = predictionModel(LogisticRegression())
# SVM
svm_pred = predictionModel(SVC())
# kNN
knn_pred_4 = predictionModel(KNeighborsClassifier(n_neighbors=4))
# Random Forest
rf_pred = predictionModel(RandomForestClassifier(n_estimators=100))
# Navie Bayes
nb_pred = predictionModel(GaussianNB())

submission = pd.DataFrame({
    "PassengerId": test['PassengerId'],
    "Survived": rf_pred
})

submission.to_csv('../Data/submission_rf.csv', index=False)

