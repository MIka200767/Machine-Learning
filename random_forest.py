import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['target'] = iris.target

x = df.drop('target', axis=1)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.8)

model = RandomForestClassifier(n_estimators=10)
model.fit(X_train,y_train)
scores = model.score(X_test,y_test)

### confusion matrix ###

y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
print(cm)