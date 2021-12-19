#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report , f1_score ,  precision_score, recall_score, roc_curve , accuracy_score, roc_auc_score
from sklearn import svm
import imblearn.over_sampling
from sklearn.neighbors import KNeighborsClassifier
#%%
df = pd.read_csv("data/clean_data.csv")
#%%
x = df[['DISBURSED_AMOUNT', 'ASSET_COST', 'PRI_NO_OF_ACCTS',
           'PRI_ACTIVE_ACCTS', 'PRI_OVERDUE_ACCTS', 'PRI_CURRENT_BALANCE',
           'PRI_DISBURSED_AMOUNT', 'PRIMARY_INSTAL_AMT',
           'NEW_ACCTS_IN_LAST_SIX_MONTHS',
           'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS', 'NO_OF_INQUIRIES',
           'LOANEE_DOB_DAYS', 'DISBURSAL_DATE_DAYS',
           'LTA', 'EMPLOYMENT_TYPE_Self employed',
           'PERFORM_CNS_DESC_Low', 'PERFORM_CNS_DESC_Medium',
           'PERFORM_CNS_DESC_High', 'PERFORM_CNS_DESC_Very high']]
y = df["LOAN_DEFAULT"]

#%%
print(df["DISBURSAL_DATE_DAYS"].head())

#%%
X_train , X_vald , y_train , y_vald = train_test_split(x , y , test_size = 0.2, random_state=42)
#%%
# setup for the ratio argument of RandomOverSampler initialization
n_pos_train = np.sum(y_train == 1)
n_neg_train = np.sum(y_train == 0)
ratio = {1 : n_pos_train * 7, 0 : n_neg_train}
# randomly oversample positive samples: create 4x as many
ROS_train = imblearn.over_sampling.RandomOverSampler( sampling_strategy = ratio,  random_state=7)
X_train_rs, y_train_rs = ROS_train.fit_resample(X_train, y_train)

scaler = StandardScaler()
scaler.fit(X_train_rs)

#%%
# Best params:  {'C': 15.194810126582277, 'penalty': 'l1', 'solver': 'saga'}
# Best params:  {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs' }
# {'max_depth': 25, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 250 }
# Best params:  {'metric': 'manhattan', 'n_neighbors': 22, 'weights': 'distance'}
lr_model = LogisticRegression(C=0.01, penalty='l1', solver='saga' ,  n_jobs=-1)
rf_model = RandomForestClassifier(n_estimators = 100  , max_depth = 20 , min_samples_split = 10 , min_samples_leaf = 2 ,  n_jobs=-1 )
# knn_model = KNeighborsClassifier( metric = 'manhattan', n_neighbors = 22, weights = 'distance' )
#%%
model = VotingClassifier([
    ('lr' , lr_model) ,
    ('rf' , rf_model) ,
    # ('knn' , knn_model )
    ], voting='hard', n_jobs=-1 )

model.fit(scaler.transform(X_train_rs), y_train_rs)
#%%
print(model.score(scaler.transform(X_train), y_train))
print(model.score(scaler.transform(X_vald), y_vald))
y_train_predict = model.predict(scaler.transform(X_train))
y_vald_predict = model.predict(scaler.transform(X_vald))

print(f"Traing F1 { f1_score(y_train , y_train_predict) }")
print(f"Val F1 { f1_score(y_vald , y_vald_predict)} ")
print(f"Traing precision_score { precision_score(y_train , y_train_predict) }")
print(f"Val precision_score { precision_score(y_vald , y_vald_predict)} ")
print(f"Traing recall_score { recall_score(y_train , y_train_predict) }")
print(f"Val recall_score { recall_score(y_vald , y_vald_predict)} ")

#%%
# model = LogisticRegression(C=0.01, penalty='l1', solver='saga' ,  n_jobs=-1)
model = RandomForestClassifier(n_estimators = 400  , max_depth = 15 , min_samples_split = 5 , min_samples_leaf = 2 ,  n_jobs=-1 )
model.fit(scaler.transform(X_train_rs), y_train_rs)
print(model.score(scaler.transform(X_train_rs), y_train_rs))
print(model.score(scaler.transform(X_vald), y_vald))
y_train_predict = model.predict(scaler.transform(X_train_rs))
y_vald_predict = model.predict(scaler.transform(X_vald))
#
# print(f"Traing F1 { f1_score(y_train_rs , y_train_predict) }")
# print(f"Val F1 { f1_score(y_vald , y_vald_predict)} ")
# print(f"Traing precision_score { precision_score(y_train , y_train_predict) }")
# print(f"Val precision_score { precision_score(y_vald , y_vald_predict)} ")
# print(f"Traing recall_score { recall_score(y_train , y_train_predict) }")
# print(f"Val recall_score { recall_score(y_vald , y_vald_predict)} ")


