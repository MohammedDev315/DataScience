#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso , LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("data/clean_data.csv")
print(df.columns)
#%%
x = df.drop("LOAN_DEFAULT" , axis = 1)
y = df["LOAN_DEFAULT"]

X_train , X_vald , y_train , y_vald = train_test_split(x , y , test_size = 0.2, random_state=42)
X_train_sub_set , X_vald_sub_set , y_train_sub_set , y_vald_sub_set = train_test_split(x[0:1000], y[0:1000] , test_size = 0.2, random_state=42)
#%%
param_grid = {
      'C' : np.logspace(-2, 2, 10),
      'penalty' : ['l1', 'l2'],
}
scaler = StandardScaler()
scaler.fit(X_train_sub_set)
X_train_scaled = scaler.transform(X_train_sub_set)

model = LogisticRegression()
model.fit(X_train_scaled, y_train_sub_set)

grid = GridSearchCV(model , param_grid , scoring='accuracy' )
grid.fit(X_train_sub_set , y_train_sub_set)
# view the complete results
print("grid_cv" , grid.cv_results_)
# examine the best model
print("Best params: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)
print("Best score: ", grid.best_score_)
#%%
scaler = StandardScaler()
scaler.fit(X_train)
sel_features = SelectFromModel(LogisticRegression(C=0.01, penalty='l2'))
sel_features.fit(scaler.transform(X_train), y_train)
selected_feat = X_train.columns[(sel_features.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print(selected_feat)
#%%



