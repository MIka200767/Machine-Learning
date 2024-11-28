import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv('titanic.csv')
model = tree.DecisionTreeClassifier()

df.drop(['Cabin', 'Embarked', 'SibSp', 'Parch', 'PassengerId','Ticket','Name'], axis=1, inplace=True)

inputs = df.drop('Survived', axis=1)
target = df.Survived

inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.Age = inputs.Age.astype(int)

le = LabelEncoder()

inputs['Sex_n'] = le.fit_transform(inputs['Sex'])
inputs.drop('Sex', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

model.fit(X_train,y_train)
score = model.score(X_test,y_test)

input_data = pd.DataFrame([[1, 90, 35.0000, 0]], columns=X_train.columns)

prediction = model.predict(input_data)
print(prediction)