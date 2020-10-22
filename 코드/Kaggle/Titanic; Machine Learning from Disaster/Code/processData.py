import pandas as pd

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

title_mapping = {"Mr":0, "Miss":1, "Mrs":2,
                 "Master":3, "Dr":3, "Rev":3, "Col":3, "Ms":3, "Mile":3, "Major":3,
                 "Lady":3, "Capt":3, "Sir":3, "Don":3, "Mme":3, "Jonkheer":3, "Countess":3}

for dataSet in train_test_data:
    dataSet['Title'] = dataSet['Title'].map(title_mapping)

print('TRAIN head')
print(train.head())
print('####################################################################')


