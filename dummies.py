import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)

# excercice 
#1)  predict price mercedes that is 4yr old with mileage 45000
#2) predict price of BMW X5 that is 7yr old with mileage 86000
#3) Score (accuracy) of the model 

df = pd.read_csv('carprices.csv') 

mod = LinearRegression()


df = pd.get_dummies(df, columns=['Car Model'], drop_first=True)

X = df.drop('Sell Price($)', axis=1)

y = df['Sell Price($)']

mod.fit(X,y)
predmers = mod.predict([[45000,4,1,0 ]])
predbmw = mod.predict([[86000, 7, 0, 1]])
score = mod.score(X,y)
print(score)