#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
import imblearn.over_sampling
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df_X = pd.read_csv('data/instacart_df_X_features.csv')

np.random.seed(42)
total_users = df_X['user_id'].unique()
test_users = np.random.choice(total_users, size=int(total_users.shape[0] * .20))

df_X_tr, df_X_te = df_X[~df_X['user_id'].isin(test_users)], df_X[df_X['user_id'].isin(test_users)]

y_tr, y_te = df_X_tr['in_cart'], df_X_te['in_cart']
X_tr, X_te = df_X_tr.drop(['product_id', 'user_id', 'latest_cart', 'in_cart'], axis=1), \
             df_X_te.drop(['product_id', 'user_id', 'latest_cart', 'in_cart'], axis=1)

#%%
# setup for the ratio argument of RandomOverSampler initialization
n_pos = np.sum(y_tr == 1)
n_neg = np.sum(y_tr == 0)
ratio = {1: n_pos * 4, 0: n_neg}

ROS = imblearn.over_sampling.RandomOverSampler(sampling_strategy=ratio, random_state=42)
X_tr_rs, y_tr_rs = ROS.fit_resample(X_tr, y_tr)
lr = LogisticRegression(solver='liblinear')
lr.fit(X_tr, y_tr)

print('Simple Logistic Regression; Test F1: %.3f, Test AUC: %.3f' % \
      (f1_score(y_te, lr.predict(X_te)), roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])))

lr_os = LogisticRegression(solver='liblinear')
lr_os.fit(X_tr_rs, y_tr_rs)
print('Logistic Regression on Oversampled Train Data; Test F1: %.3f, Test AUC: %.3f' % \
      (f1_score(y_te, lr_os.predict(X_te)), roc_auc_score(y_te, lr_os.predict_proba(X_te)[:, 1])))

#%%
smote = imblearn.over_sampling.SMOTE(sampling_strategy=ratio, random_state=42)
X_tr_smote, y_tr_smote = smote.fit_resample(X_tr, y_tr)
lr_smote = LogisticRegression(solver='liblinear')
lr_smote.fit(X_tr_smote, y_tr_smote)
print('Logistic Regression on SMOTE Train Data; Test F1: %.3f, Test AUC: %.3f' % \
      (f1_score(y_te, lr_smote.predict(X_te)), roc_auc_score(y_te, lr_smote.predict_proba(X_te)[:, 1])))

#%%
lr = LogisticRegression(solver='liblinear')
lr_balanced = LogisticRegression(class_weight='balanced', solver='liblinear')
lr_4x = LogisticRegression(class_weight={1 : 4, 0 : 1}, solver='liblinear')
lr = LogisticRegression(solver='liblinear')
lr.fit(X_tr, y_tr)
print('Normal Logistic Regression Test F1: %.3f, Test AUC: %.3f' % \
      (f1_score(y_te, lr.predict(X_te)), roc_auc_score(y_te, lr.predict_proba(X_te)[:,1])))
lr_balanced.fit(X_tr, y_tr)
print('Balanced class weights Logistic Regression Test F1: %.3f, Test AUC: %.3f' % \
      (f1_score(y_te, lr_balanced.predict(X_te)), roc_auc_score(y_te, lr_balanced.predict_proba(X_te)[:,1])))
lr_4x.fit(X_tr, y_tr)
print('4:1 class weights Logistic Regression Test F1: %.3f, Test AUC: %.3f' % \
      (f1_score(y_te, lr_4x.predict(X_te)), roc_auc_score(y_te, lr_4x.predict_proba(X_te)[:,1])))

#%%
X_val, y_val = X_te, y_te  # explicitly calling this validation since we're using it for selection

thresh_ps = np.linspace(.10, .50, 1000)
model_val_probs = lr.predict_proba(X_val)[:, 1]  # positive class probs, same basic logistic model we fit in section 2

f1_scores, prec_scores, rec_scores, acc_scores = [], [], [], []
for p in thresh_ps:
    model_val_labels = model_val_probs >= p
    f1_scores.append(f1_score(y_val, model_val_labels))
    prec_scores.append(precision_score(y_val, model_val_labels))
    rec_scores.append(recall_score(y_val, model_val_labels))
    acc_scores.append(accuracy_score(y_val, model_val_labels))

plt.plot(thresh_ps, f1_scores)
plt.plot(thresh_ps, prec_scores)
plt.plot(thresh_ps, rec_scores)
plt.plot(thresh_ps, acc_scores)

plt.title('Metric Scores vs. Positive Class Decision Probability Threshold')
plt.legend(['F1', 'Precision', 'Recall', 'Accuracy'], bbox_to_anchor=(1.05, 0), loc='lower left')
plt.xlabel('P threshold')
plt.ylabel('Metric score')

best_f1_score = np.max(f1_scores)
best_thresh_p = thresh_ps[np.argmax(f1_scores)]

print('Logistic Regression Model best F1 score %.3f at prob decision threshold >= %.3f'
      % (best_f1_score, best_thresh_p))
plt.show()