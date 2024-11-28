import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', 200)


wine = load_wine()

df = pd.DataFrame(wine.data,columns=wine.feature_names)
df['target'] = wine.target
x = df.drop('target', axis=1)
y = df.target
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.8,random_state=100)

modelNB = GaussianNB()
modelNB.fit(X_train,y_train)
sc_nb = modelNB.score(X_test,y_test)
print(sc_nb)

modelMLT = MultinomialNB()
modelMLT.fit(X_train,y_train)
sc_mlt = modelMLT.score(X_test,y_test)
print(sc_mlt)
