#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso , LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import imblearn.over_sampling

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
X_train , X_vald , y_train , y_vald = train_test_split(x , y , test_size = 0.2, random_state=42)
#%%
# setup for the ratio argument of RandomOverSampler initialization
n_pos_train = np.sum(y_train == 1)
n_neg_train = np.sum(y_train == 0)
ratio = {1 : n_pos_train * 5, 0 : n_neg_train}
# randomly oversample positive samples: create 4x as many
ROS_train = imblearn.over_sampling.RandomOverSampler( sampling_strategy = ratio,  random_state=7)
X_train_rs, y_train_rs = ROS_train.fit_resample(X_train, y_train)

scaler = StandardScaler()
scaler.fit(X_train_rs)

#%%
param_grid = {
      "C" : np.linspace(0.01,30,80),
      'solver' : ['lbfgs' , 'saga'],
      'penalty' : ['l1' , 'l2']
}

#%%
model = LogisticRegression()
model.fit(X_train_rs , y_train_rs)


#%%
grid = GridSearchCV(model , param_grid , scoring='f1' , n_jobs=-1)
grid.fit(X_train_rs , y_train_rs)
print("grid_cv" , grid.cv_results_)
# examine the best model
print("Best params: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)
print("Best score: ", grid.best_score_)



#%%
# model = LogisticRegression(C=0.414, penalty='l1', solver='saga' ,n_jobs=-1)
# model.fit(X_train_scaled , y_train)
# print(model.score(scaler.transform(X_train) , y_train))
# print(model.score(scaler.transform(X_vald) , y_vald))
# y_train_predict = model.predict(X_train)
# y_vald_predict = model.predict(X_vald)
# print(classification_report(y_train , y_train_predict))
# print(classification_report(y_vald , y_vald_predict))
