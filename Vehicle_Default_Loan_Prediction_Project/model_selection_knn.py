#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
X_train_sub_set , X_vald_sub_set , y_train_sub_set , y_vald_sub_set = train_test_split(x[0:100], y[0:100] , test_size = 0.2, random_state=42)
#%%
colname = ['DISBURSED_AMOUNT', 'ASSET_COST', 'PRI_NO_OF_ACCTS',
           'PRI_ACTIVE_ACCTS', 'PRI_OVERDUE_ACCTS', 'PRI_CURRENT_BALANCE',
           'PRI_DISBURSED_AMOUNT', 'PRIMARY_INSTAL_AMT',
           'NEW_ACCTS_IN_LAST_SIX_MONTHS',
           'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS', 'NO_OF_INQUIRIES',
           'LOANEE_DOB_DAYS', 'DISBURSAL_DATE_DAYS',
           'LTA', 'EMPLOYMENT_TYPE_Self employed',
           'PERFORM_CNS_DESC_Low', 'PERFORM_CNS_DESC_Medium',
           'PERFORM_CNS_DESC_High', 'PERFORM_CNS_DESC_Very high']

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

#List Hyperparameters that we want to tune.
leaf_size = list(range(10,40))
n_neighbors = list(range(5,25))
p=[1,2]
n_jobs = -1

#Convert to dictionary
param_grid = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

model = KNeighborsClassifier()
model.fit(X_train_scaled, y_train)

grid = GridSearchCV(model , param_grid , scoring='f1' , n_jobs = -1)
grid.fit(X_train , y_train)
# view the complete results
print("grid_cv" , grid.cv_results_)
# examine the best model
print("Best params: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)
print("Best score: ", grid.best_score_)
#%%
sel_features = SelectFromModel(KNeighborsClassifier(leaf_size = 5 ,n_neighbors = 3 , p=2 , n_jobs = -1))
sel_features.fit(scaler.transform(X_train), y_train)
selected_feat = X_train.columns[(sel_features.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print(selected_feat)




