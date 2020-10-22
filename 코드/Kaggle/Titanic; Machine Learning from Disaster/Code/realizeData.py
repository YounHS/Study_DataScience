import pandas as pd

train = pd.read_csv('../Data/train.csv')
test = pd.read_csv('../Data/test.csv')

##################################################################
# 데이터 딕셔너리
# survived : 생존=1, 죽음=0
# pclass : 승객 등급. 1등급=1, 2등급=2, 3등급=3
# sibsp : 함께 탑승한 형제 또는 배우자 수
# parch : 함께 탑승한 부모 또는 자녀 수
# ticket : 티켓 번호
# fare : 요금
# cabin : 선실 번호
# embarked : 탑승장소 S=Southhampton, C=Cherbourg, Q=Queenstown
##################################################################

print('TRAIN DATA')
print(train.head())

print('TRAIN SHAPE:', train.shape)
print('TRAIN info: ')
print(train.info())  # Age data num: 714, Cabin data num: 204

print('##################################################################')
print('TEST DATA')
print(test.head())

print('TEST SHAPE:', test.shape)
print('TEST info: ')
print(test.info())  # Age data num: 714, Cabin data num: 204

