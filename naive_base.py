import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)

model = GaussianNB()

df =  pd.read_csv('titanic.csv')

df.drop(['PassengerId','Pclass', 'Name','Parch', 'Ticket','Embarked','Cabin'], axis=1, inplace=True)
df.Age = df.Age.fillna(df.Age.mean())

inputs = df.drop('Survived', axis=1)
target = df.Survived

dummies = pd.get_dummies(inputs, columns=['Sex'])
inputs = pd.concat([inputs,dummies],axis=1)
inputs.drop(['Sex'], axis=1, inplace=True)
print(inputs) 