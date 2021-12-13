#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
#%%
df = pd.read_csv("data/MIMIC_Data_small.csv")
print(df.columns)

sns.heatmap( df.isnull() ,  yticklabels = False , cbar = False , cmap = "viridis" )
plt.show()
sns.countplot(x = 'require_surgery_flag' , hue='Sex' ,  data = df)
plt.show()
#%%
x = df.drop("require_surgery_flag" , axis = 1)
y = df["require_surgery_flag"]
X_train , X_test , y_train , y_test = train_test_split(x , y ,test_size=.2 , random_state=42)
lg = LogisticRegression()
lg.fit(X_train , y_train)
print(lg.score(X_train , y_train)) #0.8950
#%%
pro_need_surgary = lg.predict_proba(X_train)[:, 1].mean()
pro_not_need_surgary = lg.predict_proba(X_train)[:, 0].mean()
print(pro_need_surgary)
print(pro_not_need_surgary)
#%%
sns.pairplot(df , hue = 'require_surgery_flag' )
plt.show()
#%%
mask1 = (X_train.loc[: , 'resprate_mean'] > 30) & (X_train.loc[: , 'resprate_mean'] < 32)
print(f"Need Surgary {lg.predict_proba(X_train[mask1])[:, 1].mean()} ")
print(f"Not Need Surgary {lg.predict_proba(X_train[mask1])[:, 0].mean()} ")
mask1 = (X_train.loc[: , 'resprate_mean'] > 16) & (X_train.loc[: , 'resprate_mean'] < 18)
print(f"Need Surgary {lg.predict_proba(X_train[mask1])[:, 1].mean()} ")
print(f"Not Need Surgary {lg.predict_proba(X_train[mask1])[:, 0].mean()} ")
mask1 = (X_train.loc[: , 'resprate_mean'] > 6) & (X_train.loc[: , 'resprate_mean'] < 12)
print(f"Need Surgary {lg.predict_proba(X_train[mask1])[:, 1].mean()} ")
print(f"Not Need Surgary {lg.predict_proba(X_train[mask1])[:, 0].mean()} ")
#%%
for col in X_train.columns:
    print(col)
    X_tr = X_train[[col]]
    X_te = X_test[[col]]
    lr_model = LogisticRegression()
    lr_model.fit(X_tr , y_train)
    y_porb_pred_test = lr_model.predict_proba(X_te)[: , 1]
    print(log_loss(y_test , y_porb_pred_test))

#%%
lr_model_all = LogisticRegression(C=.1)
lr_model_all.fit(X_train , y_train)
y_porb_pred_test = lr_model_all.predict_proba(X_test)[:,1]
y_porb_pred_train = lr_model_all.predict_proba(X_train)[:,1]
print(log_loss(y_test , y_porb_pred_test))
print(log_loss(y_train, y_porb_pred_train))
print(lr_model_all.score(X_train , y_train))
print(lr_model_all.score(X_test , y_test))
#%%


