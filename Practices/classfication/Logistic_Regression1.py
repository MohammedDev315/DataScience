#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df_housing = pd.read_csv('ny_sf_apt.csv')
df_housing['location'] = df_housing['in_sf'].apply(lambda x: 'SF' if x == 1 else 'NY')
X_train, X_test, y_train, y_test = train_test_split(df_housing.iloc[:, 1:], df_housing.iloc[:, 0],
                                                    test_size = 0.2, random_state=42)

train_df = X_train.copy()
train_df['in_sf'] = y_train


lm1 = LogisticRegression(C=1000) # setting C very high essentially removes regularization
lm1.fit(X_train[['elevation']], y_train)
print(lm1.score(X_train[['elevation']], y_train))
print(lm1.score(X_test[['elevation']], y_test))

#%%
lm1.predict_proba([[73],[23],[25]])[:,1] # 2nd column <-> prob class 1

#%%
lm1.coef_, lm1.intercept_

#%%
keep_sf_mask = ((train_df['location'] == 'SF') & (train_df['elevation'] > 24))
keep_ny_mask = ((train_df['location'] == 'NY') & (train_df['elevation'] < 24))
cheat_df = train_df[keep_sf_mask | keep_ny_mask]

lm2 = LogisticRegression(C=1000)
lm2.fit(cheat_df[['elevation']], cheat_df['in_sf'])
print(train_df.head())

#%%
std_scale = StandardScaler()

X_train = train_df[['elevation', 'price_per_sqft']]
X_train_scaled = std_scale.fit_transform(X_train)

lm3 = LogisticRegression()
lm3.fit(X_train_scaled, y_train)

y_predict = lm3.predict(X_train_scaled)
print(lm3.score(X_train_scaled, y_train))

