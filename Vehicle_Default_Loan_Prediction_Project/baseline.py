#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/clean_data.csv")
print(df.columns)
#%%
x = df.drop("LOAN_DEFAULT" , axis = 1)
y = df["LOAN_DEFAULT"]

X_train , X_vald , y_train , y_vald = train_test_split(x , y , test_size = 0.2, random_state=42)
#%%
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train , y_train)
print(f"Knn Train{model.score(X_train , y_train)}")
print(f"Knn Val{model.score(X_vald , y_vald)}")
y_pred = model.predict(X_vald)
print(classification_report(y_vald, y_pred))
cf_matrix = confusion_matrix(y_vald, y_pred)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.show()
#%%
model = LogisticRegression()
model.fit(X_train , y_train)
print(f"Logistic Train{model.score(X_train , y_train)}")
print(f"Logistic Val{model.score(X_vald , y_vald)}")
y_pred = model.predict(X_vald)
print(classification_report(y_vald, y_pred))
cf_matrix = confusion_matrix(y_vald, y_pred)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.show()





