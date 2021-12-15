#%%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

df_train = pd.read_csv("data/wookiee-train.csv")
df_test = pd.read_csv("data/wookiee-test.csv")
print(df_train.columns)
#%%
X_train = df_train.drop(["wookieecolor" , "Unnamed: 0"] , axis=1)
y_train = df_train["wookieecolor"]
X_test = df_test.drop(["wookieecolor" , "Unnamed: 0"] , axis=1)
y_test = df_test["wookieecolor"]
#%%
lr_model = LogisticRegression()
knn_model = KNeighborsClassifier()
rf_model = RandomForestClassifier()
et_model = ExtraTreesClassifier()

model = VotingClassifier([
    ('lr' , lr_model) ,
    ('knn' , knn_model) ,
    ('rf' , rf_model) ,
    ('et' , et_model) ,
    ], voting='hard', n_jobs=-1 )
#%%
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

#%%
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))