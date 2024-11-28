import pandas as pd
import os
import kagglehub
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_columns', 200)


path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
file = os.path.join(path, 'heart.csv')
df = pd.read_csv(file)

for col in df.select_dtypes(include='number').columns:
    z_score = (df[col] - df[col].mean()) / df[col].std()
    df = df[(z_score > -3) & (z_score < 3)]

df = pd.get_dummies(df, columns=['Sex','ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)

for i in df.select_dtypes(include='bool').columns:
    df[i] = df[i].astype(int)

x = df.drop('HeartDisease', axis=1)
y = df.HeartDisease

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x_scaled,y,test_size=0.8,random_state=10)

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
score_l = (log_reg.score(X_test,y_test))*100
print(round(score_l, 1),"% :Logistic Regression")

bag = BaggingClassifier(
    n_estimators=100, 
    max_samples=0.8,
    oob_score=True,
    random_state=0)

bag.fit(X_train,y_train)
score_bag = (bag.oob_score_)*100
print(round(score_bag, 1),"% :Bagging")

score_tree = cross_val_score(DecisionTreeClassifier(), x, y, cv=5)
print(score_tree.mean(),":Decision Tree")

score = cross_val_score(RandomForestClassifier(n_estimators=20), x, y, cv=5)
print(score.mean(), ":Random Forest")