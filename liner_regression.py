import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)


# df = pd.read_csv('canada_per_capita_income.csv', index_col=False)
# df = df.rename(columns={'per capita income (US$)': 'Income', 'year': 'Year'})

# # sns.scatterplot(data=df, x='Year', y='Income')
# # plt.show()

# reg = linear_model.LinearRegression()

# reg.fit(df[['Year']], df.Income)

# p = reg.predict(pd.DataFrame([[2020]], columns=['Year']))
# # y = m*x+b

# sns.scatterplot(data=df, x='Year', y='Income')
# plt.plot(df.Year, reg.predict(df[['Year']]), color='blue' )
# plt.show()


###### Multiple independant variables #######
from word2number import w2n

df = pd.read_csv('hiring.csv')
reg = linear_model.LinearRegression()
df.experience = df.experience.fillna('zero')
df.experience = df.experience.apply(w2n.word_to_num)

for i in ['test_score(out of 10)']:
    df[i] = df[i].fillna(df[i].mean()).round()

reg.fit(df[['experience',  'test_score(out of 10)',  'interview_score(out of 10)']], df['salary($)'])
predict = reg.predict([[12,10,10]])
print(predict)

