# %%
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import ssl
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import xgboost as xgb

ssl._create_default_https_context = ssl._create_unverified_context

housing_dataset = fetch_california_housing()
X = pd.DataFrame(housing_dataset.data)
X.columns = housing_dataset.feature_names
y = housing_dataset.target
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2018)
linrg = LinearRegression()
linrg.fit(X_train, y_train)
print(linrg.score(X_train, y_train))
print(linrg.score(X_test, y_test))
predeicted = linrg.predict(X_train)


# %%
def fit(X, y, n_estimators, max_depth=3, learining_rate=1):
    C = np.mean(y)
    estimators = []
    resids = y - C
    for _ in range(n_estimators):
        est = DecisionTreeRegressor(max_depth=max_depth)
        est.fit(X, resids)
        resids -= learining_rate * est.predict(X)
        estimators.append(est)
    return estimators, C


def predict(estimators, C, X, learing_rate=1):
    y_pred = np.repeat(C, len(X))
    for est in estimators:
        y_pred += learing_rate * est.predict(X)
    return y_pred

#%%
estimators , C = fit(X_train , y_train , n_estimators=100 , max_depth=3 , learining_rate=.71)
y_pred = predict(estimators , C ,X_test , learing_rate=0.71)
print(r2_score(y_test , y_pred))
#%%

#%%
class IAmTheOneWhoBoosts():
    def __init__(self, n_estimators=10, max_depth=3, learning_rate=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
    def fit(self, X, y):
        self.C = np.mean(y)
        self.estimators = []
        resids = y - self.C
        for _ in range(self.n_estimators):
            est = DecisionTreeRegressor(max_depth=self.max_depth)
            est.fit(X, resids)
            resids -= self.learning_rate * est.predict(X)
            self.estimators.append(est)
    def predict(self, X):
        return self.C + np.sum([self.learning_rate * est.predict(X) \
                                for est in self.estimators], axis=0)
    def score(self, X, y):
        return r2_score(y, self.predict(X))

#%%
booster = IAmTheOneWhoBoosts(n_estimators=100 , max_depth=3 , learining_rate=.71)
booster.fit(X_train , y_train)
booster.score(X_train , y_train)
#%%
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=100 , max_depth=3 , learning_rate=.71)
gb.fit(X_train , y_train)
y_pred = gb.predict(X_train)
print(r2_score(y_test , y_pred))