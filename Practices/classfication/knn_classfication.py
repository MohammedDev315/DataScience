#%%
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

df_train = pd.read_csv("data/wookiee-train.csv")
df_test = pd.read_csv("data/wookiee-test.csv")
print(df_train.columns)
#%%
print(df_train.shape)
#%%
X_train = df_train.drop(["wookieecolor" , "Unnamed: 0"] , axis=1)
y_train = df_train["wookieecolor"]
X_test = df_test.drop(["wookieecolor" , "Unnamed: 0"] , axis=1)
y_test = df_test["wookieecolor"]
#%%
print(X_train.columns)
#%%
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train , y_train)
print(knn.score(X_train , y_train))
#%%
k_range = list(range(1, 100))
print(k_range)
#%%
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn , param_grid , cv=5 , scoring='accuracy' )
grid.fit(X_train , y_train)
# view the complete results
print("grid_cv" , grid.cv_results_)
# examine the best model
print("Best params: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)
print("Best score: ", grid.best_score_)
#%%
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train , y_train)
print("Train : " , knn.score(X_train , y_train))
print("Test : " , knn.score(X_test , y_test))
#%%
print(knn.predict([[-3.410692 , 0.85440 , 0.228154]]))