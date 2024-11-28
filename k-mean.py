import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df.drop(['petal length (cm)','petal width (cm)'], axis=1, inplace=True)

# plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'])
# plt.show()

scaler = MinMaxScaler()
scaler.fit(df[['sepal length (cm)']])
df['sepal length (cm)'] = scaler.transform(df[['sepal length (cm)']])
scaler.fit(df[['sepal width (cm)']])
df['sepal width (cm)'] = scaler.transform(df[['sepal width (cm)']])

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df)
df['flower'] = y_predicted
df0 = df[df.flower==0]
df1 = df[df.flower==1]
df2 = df[df.flower==2]
# df3 = df[df.flower==3]


plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='blue')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='green')
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='yellow')
# plt.scatter(df3['sepal length (cm)'],df3['sepal width (cm)'],color='yellow')
plt.show()

