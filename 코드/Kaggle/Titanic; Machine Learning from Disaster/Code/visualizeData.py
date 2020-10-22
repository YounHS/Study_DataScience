import matplotlib.pyplot as plt
import seaborn as sb
import realizeData as rd

######################################
# 범주형 features에 대한 막대차트
# pclass
# sex
# sibsp
# parch
# embarked
# cabin
######################################

sb.set()


def bar_chart(feature):
    survived = rd.train[rd.train['Survived'] == 1][feature].value_counts()
    dead = rd.train[rd.train['Survived'] == 0][feature].value_counts()
    df = rd.pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.show()


print(bar_chart('Sex'))

print(bar_chart('Pclass'))

print(bar_chart('SibSp'))

print(bar_chart('Parch'))

print(bar_chart('Embarked'))
