import pandas as pd
import kagglehub
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pd.set_option('display.max_columns', 200)


path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
file = os.path.join(path, 'heart.csv')
df = pd.read_csv(file)

for col in df.select_dtypes(include='number').columns:
    z_score = (df[col] - df[col].mean()) / df[col].std()
    df = df[(z_score <= 3) & (z_score >= -3)]

df = pd.get_dummies(df, columns=['Sex','ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)

for col in df.select_dtypes(include=['bool']).columns:
    df[col] = df[col].astype(int)

print(df.RestingECG_ST.unique())

inputs = df.drop('HeartDisease', axis=1)
target = df.HeartDisease

scaler = StandardScaler()
X_scaled = scaler.fit_transform(inputs)

X_train, X_test, y_train, y_test =  train_test_split(X_scaled, target, test_size=0.2, random_state=10)

logistic = LogisticRegression()
logistic.fit(X_train,y_train)
logistic_score = logistic.score(X_test,y_test)
print(logistic_score*100,"% :Logistic Regression")

rand_forest = RandomForestClassifier(n_estimators=10)
rand_forest.fit(X_test,y_test)
rand_forest_score = rand_forest.score(X_test, y_test)
print(rand_forest_score*100,"% :Random Forest")

pca = PCA(0.90)
x_pca = pca.fit_transform(inputs)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(x_pca, target, test_size=0.2, random_state=30)
model_pca = LogisticRegression()
model_pca.fit(X_train_pca, y_train_pca)
pca_log_score = model_pca.score(X_test_pca, y_test_pca)
print(pca_log_score*100,'% :Logistic Regression using PCA')