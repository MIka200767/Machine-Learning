import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import numpy as np
pd.set_option('display.max_columns', 400)

##example no real dataset###
df = pd.read_csv('titanic.csv')

df.drop(['Cabin','Ticket', 'Name', ],axis=1,inplace=True)

df.Age = df.Age.fillna(df.Age.mean())

df['Sex_n'] = df.Sex.map({'male':1,'female':0})
df.drop('Sex',axis=1,inplace=True)
df = pd.get_dummies(df, drop_first=True)
df['Embarked_Q_numeric'] = df['Embarked_Q'].astype(int)
df['Embarked_S_numeric'] = df['Embarked_S'].astype(int)
df.drop(['Embarked_Q',  'Embarked_S'], axis=1, inplace=True)

x = df.drop('Survived', axis=1)
y = df.Survived

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.8)

model = LinearRegression().fit(X_train,y_train)
score_test = model.score(X_test,y_test)
score_train = model.score(X_train,y_train)

lasor_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasor_reg.fit(X_train,y_train)
