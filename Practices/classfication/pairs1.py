import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# print(train_df.head())
# print(train_df.columns)
# print(train_df["Embarked"].value_counts())
# print(train_df["Embarked"].describe())
# print(train_df.info())
# print(train_df.isnull().sum(axis = 0))
# train_empty_age = train_df.loc[train_df.loc[:, "Age"].isnull()]


train = pd.read_csv("wookiee-train.csv").drop('Unnamed: 0' , axis = 1)
test = pd.read_csv("wookiee-test.csv").drop('Unnamed: 0' , axis = 1)


x_train_1 = train.drop('wookieecolor', axis = 1)
y_train = train["wookieecolor"]
x_test = test.drop('wookieecolor' , axis = 1)
y_test = test["wookieecolor"]

ss = StandardScaler()
ss.fit(x_train_1)

scaled_x_train = ss.transform(x_train_1)
scaled_x_test = ss.transform(x_test)

knn = KNeighborsClassifier()
knn.fit(scaled_x_train , y_train)
train_rsult = knn.score(scaled_x_train , y_train)
test_rsult = knn.score(scaled_x_test , y_test)
# print(train_rsult)
# print(test_rsult)

k_range = list(range(1, 100))
# param_grid = dict('n_neighbors' , k_range)
parm = {'n_neighbors' : list(range(50)) }
grid = GridSearchCV(knn, param_grid = parm, cv=10)
grid.fit(scaled_x_train, y_train)

knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')
knn.fit(scaled_x_train, y_train)

# make a prediction on out-of-sample data
print(knn.predict([[3, 5, 4]]))
print(knn.predict([[3, 5, 4]]))




