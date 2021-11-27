import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
# print(train_data.head())
# print(train_data.info())
# print(train_data.describe())
# print(train_data.columns)
# sns.pairplot(train_data)
# sns.boxplot(x = train_data["x1"])
# sns.boxplot(x = train_data["x2"])
# sns.boxplot(x = train_data["y"])
# sns.heatmap(train_data.corr())
# plt.show()
X_train, X_test, y_train, y_test = train_test_split(train_data, test_data, test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)





