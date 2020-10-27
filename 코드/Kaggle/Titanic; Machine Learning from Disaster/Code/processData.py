import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

train = pd.read_csv('../Data/train.csv')
test = pd.read_csv('../Data/test.csv')

train_test_data = [train, test]

print(train_test_data)
print('####################################################################')

for dataSet in train_test_data:
    dataSet['Title'] = dataSet['Name'].str.extract('([A-za-z]+)\.', expand=False)

print('DATASET SHOW')
print(dataSet)
print('####################################################################')

print('TRAIN title value count')
print(train['Title'].value_counts())
print('####################################################################')

print('TEST title value count')
print(test['Title'].value_counts())
print('####################################################################')

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Ms": 3, "Mile": 3, "Major": 3,
                 "Lady": 3, "Capt": 3, "Sir": 3, "Don": 3, "Mme": 3, "Jonkheer": 3, "Countess": 3}

for dataSet in train_test_data:
    dataSet['Title'] = dataSet['Title'].map(title_mapping)

print('TRAIN head')
print(train.head())
print('####################################################################')


def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.show()


print('####################################################################')
# print(bar_chart('Title'))

# Name 컬럼은 중요한 데이터가 아니므로 삭제
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

print('####################################################################')
print(train.head())
print('####################################################################')

# 성별과 생존/죽음 관계
gender_mapping = {'male': 0, 'female': 1}
for dataSet in train_test_data:
    dataSet['Sex'] = dataSet['Sex'].map(gender_mapping)

print('####################################################################')
# print(bar_chart('Sex'))

# 나이와 생존/죽음 관계
# 결측치인 나이를 각 Title에 대한 연령의 중간값으로 대체 (Mr, Mrs, Miss, Others)
train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)

facet = sb.FacetGrid(train, hue='Survived', aspect=4)
facet.map(sb.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
sb.axes_style("darkgrid")

# 0 ~ 10 대 중반까지 생존률 높으며, 30, 70대의 사망률 높음
# plt.show()

facet = sb.FacetGrid(train, hue='Survived', aspect=4)
facet.map(sb.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
# plt.xlim(0, 20)
# plt.xlim(20, 30)
# plt.xlim(30, 40)
# plt.xlim(40, 60)
# plt.xlim(60)
plt.style.use('ggplot')

# 연령대별로 split하여 그래프 자세히 보기
# plt.show()

print(train.info())
print('####################################################################')
print(test.info())
print('####################################################################')

# feature Engineering의 Binning 기술을 사용하여 대량의 데이터를 하나의 카테고리에 담아 정보를 보다
# 명확하게 확인할 수 있게 함.
for dataSet in train_test_data:
    dataSet.loc[dataSet['Age'] <= 16, 'Age'] = 0,
    dataSet.loc[(dataSet['Age'] > 16) & (dataSet['Age'] <= 26), 'Age'] = 1,
    dataSet.loc[(dataSet['Age'] > 26) & (dataSet['Age'] <= 36), 'Age'] = 2,
    dataSet.loc[(dataSet['Age'] > 36) & (dataSet['Age'] <= 62), 'Age'] = 3,
    dataSet.loc[(dataSet['Age'] > 62), 'Age'] = 4

print(train.head())
print('####################################################################')

# print(bar_chart('Age'))

Pclass_1 = train[train['Pclass'] == 1]['Embarked'].value_counts()
Pclass_2 = train[train['Pclass'] == 2]['Embarked'].value_counts()
Pclass_3 = train[train['Pclass'] == 3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass_1, Pclass_2, Pclass_3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10, 5))

# 도시별 1,2,3등급 좌석 승객 수 보기
# plt.show()

# 상기 그래프 출력 결과 S 도시에서 탑승한 승객이 1, 2, 3등급 좌석의 비중이 모두 높으므로, Embarked 정보의 결측치는
# S로 대체하기로 함
for dataSet in train_test_data:
    dataSet['Embarked'] = dataSet['Embarked'].fillna('S')

print(train.head())
print('####################################################################')

# 추후 ML의 classfier 작업을 위해 텍스트를 숫자로 변환
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
for dataSet in train_test_data:
    dataSet['Embarked'] = dataSet['Embarked'].map(embarked_mapping)

# 티켓 가격은 클래스와 관련있을 확률이 높으며 클래스는 결측치 값이 존재하지 않으므로, 각 클래스의 티켓 가격의 가운데 값을 가격의
# 결측치 값으로 대체하기로 함
train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'), inplace=True)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'), inplace=True)

facet = sb.FacetGrid(train, hue='Survived', aspect=4)
facet.map(sb.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
# plt.xlim(0, 20)
# plt.xlim(0, 30)
# plt.xlim(0)   # 티켓의 갯수가 0이 되는 티켓의 가격
# plt.show()

# feature Engineering의 Binning 기술을 사용하여 대량의 데이터를 하나의 카테고리에 담아 정보를 보다
# 명확하게 확인할 수 있게 함.
for dataSet in train_test_data:
    dataSet.loc[dataSet['Fare'] <= 17, 'Fare'] = 0,
    dataSet.loc[(dataSet['Fare'] > 17) & (dataSet['Fare'] <= 30), 'Fare'] = 1,
    dataSet.loc[(dataSet['Fare'] > 30) & (dataSet['Fare'] <= 100), 'Fare'] = 2,
    dataSet.loc[dataSet['Fare'] > 100, 'Fare'] = 3

print(train.head())
print('####################################################################')

# Cabin 데이터
print(train.Cabin.value_counts())
print('####################################################################')

# 알파벳과 숫자의 조합은 핸들링이 어려우므로 첫 글자가 무조건 알파벳임을 확인하고,
# 첫 글자인 알파벳만 추출하여 사용키로 함.
for dataSet in train_test_data:
    dataSet['Cabin'] = dataSet['Cabin'].str[:1]

Pclass_1 = train[train['Pclass'] == 1]['Cabin'].value_counts()
Pclass_2 = train[train['Pclass'] == 2]['Cabin'].value_counts()
Pclass_3 = train[train['Pclass'] == 3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass_1, Pclass_2, Pclass_3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10, 5))

# 1등급에는 ABCDET, 2등급에는 DEF, 3등급은 EFG로 구성
# plt.show()

# 상기 결과를 classfier를 위해 매핑
cabin_mapping = {'A': 0, 'B': 0.4, 'C': 0.8, 'D': 1.2, 'E': 1.6, 'F': 2, 'G': 2.4, 'T': 2.8}
for dataSet in train_test_data:
    dataSet['Cabin'] = dataSet['Cabin'].map(cabin_mapping)

# 상기처럼 소수점을 사용하는 매핑을 feature scaling이라고 하며, ML classfier는 숫자를 사용하고 계산을 할 때
# 보통 euclidean distance를 사용함.
# 정수를 사용하면 각기 다른 컬럼 내 값들의 편차가 심해지므로 소수를 사용함.

# Cabin의 결측치는 1,2,3등급 클래스와 관련이 있기 때문에 클래스 별 cabin의 중간값을 결측치 값으로 대체하기로 함.
train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'), inplace=True)

# 가족 규모에 따른 생존/사망률
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

facet = sb.FacetGrid(train, hue='Survived', aspect=4)
facet.map(sb.kdeplot, 'FamilySize', shade=True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()

# 혼자 탑승했을 때 사망률이 높음
# plt.show()

# 상기 가족 규모를 숫자로 매핑시킴. scaling 해주는데, 가족 규모의 범위를 조금이라도 좁히기 위해서라고 사료됨.
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataSet in train_test_data:
    dataSet['FamilySize'] = dataSet['FamilySize'].map(family_mapping)

print(train.head())
print('####################################################################')

# Ticket, SibSp, Parch 데이터는 필요한 정보가 아니므로 삭제
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop('PassengerId', axis=1)

########## edit & add code ##########
train = pd.get_dummies(train)
test = pd.get_dummies(test)
########## edit & add code ##########

train_label = train['Survived']
train_data = train.drop('Survived', axis=1)
test_data = test.drop('PassengerId', axis=1).copy()

print(test_data, train_data.shape, train_label.shape)
print('####################################################################')

print(test_data.head(10))
test_data['Title'] = test_data['Title'].fillna(1.0)
print(test_data.info())
print('####################################################################')

print(train_data.head(10))
train_data['Title'] = train_data['Title'].fillna(1.0)
print(train_data.info())
print('####################################################################')

print(train_label.head(10))
print(train_data.count())
print('####################################################################')

test['Title'] = test['Title'].fillna(0.1)
print(test.info())


