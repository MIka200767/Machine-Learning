import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pylab as plt
from sklearn.neighbors import KNeighborsClassifier
pd.set_option('display.max_columns', 400)


iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['target'] = iris.target

inputs = df.drop('target',axis=1)
taregt = df.target

df0 = df[df['target']==0]
df1 = df[df['target']==1]

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(inputs, taregt, test_size=0.2, random_state=10)

#### K-nearest ####

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train,y_train)
sc = knn.score(X_test,y_test)
pred = knn.predict([[7.0,5.2,4.2,1.9]])


from sklearn.metrics import confusion_matrix

y_pred = knn.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)
print(matrix)