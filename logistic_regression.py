import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)


df = pd.read_csv('HR_comma_sep.csv')

# plt.scatter(df.salary, df.left, marker='+', color='red')
# plt.show()

correlation = df[['satisfaction_level', 'left']].corr()

X_train, X_test, y_train, y_test = train_test_split(df[['satisfaction_level']], df['left'], test_size=0.1)

model = LogisticRegression()
model.fit(X_train,y_train)
p = model.predict(X_test)
print(p.tolist())
print(model.score(X_test,y_test))


