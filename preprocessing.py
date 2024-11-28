import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)


df = pd.read_excel('Life Expectancy Data.xlsx')

# OUTLIER treatment

#1) using percentile

# minp, maxp = df.Population.quantile([0.05, 0.95])

# new_df = df[(df.Population > minp) & (df.Population < maxp)]
# print(new_df.Population.shape[0])
# print(df.Population.shape[0])

#2) using Z-score, standard deviation


# for i in df.select_dtypes(include='number').columns:
#     sns.histplot(data=df, x=i)
#     plt.show()

# upper_limit = df.BMI.mean() + 3*df.BMI.std()
# lower_limit = df.BMI.mean() - 3*df.BMI.std()

# new_BMI = df[(df.BMI>lower_limit) & (df.BMI<upper_limit)]
# print(df.BMI.sort_values())

# z-score z = x - mean / std
# df['z-score'] = (df.BMI - df.BMI.mean()) / df.BMI.std()

# new_df = df[df['z-score']>-3]

###### handling MISSIGN VALUES ######

for i in ['Infant deaths', 'Alcohol', 'GDP']:
    df[i]=df[i].fillna(df[i].median())

# new_df = df.fillna({
#     'BMI': df.BMI.median(),
#     'GDP': df.GDP.mean()    
# })
# print(new_df.columns)

###### NORMALIZATION ######

## Simple Feature scaling ##

# x = x(old)/x(max)

## MIN-MAX ##

# x = (x(old) - x(min))/ x(max)-x(min)

## Z-score ##

# x = (x(old) - average) / standard deviation

########  TRAIN and TEST ########
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

plt.scatter(df['Infant deaths'], df['Alcohol'])
# plt.show()
x = df[['Alcohol', 'GDP']]
y = df['Infant deaths']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=10)

clf = LinearRegression()

clf.fit(x_train, y_train)
clf.predict(x_test)
print(clf.score(x_test,y_test))


