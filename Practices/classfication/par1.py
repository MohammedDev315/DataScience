# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

def accuracy(actual, preds):
    return np.mean(actual == preds)


def precision(actual, preds):
    tp = np.sum((actual == 1) & np.sum(preds == 1))
    fp = np.sum((actual == 0) & np.sum(preds == 1))
    return tp / (tp+fp)


def recall(actual, preds):
    tp = np.sum((actual == 1) & np.sum(preds == 1))
    fn = np.sum((actual == 1) & np.sum(preds == 0))
    return tp / (tp + fn)


def F1(actual, preds):
    p, r = precision(actual, preds), recall(actual, preds)
    return 2 * p * r / (p + r)


#mostly if donload data from sklearn, we have
bc_dataset = load_breast_cancer()
X = pd.DataFrame(bc_dataset.data)
X.columns = bc_dataset.feature_names
y = bc_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

lr = LogisticRegression(C = 1)
knn = KNeighborsClassifier(n_neighbors=10)

#%%
lr.fit(X_train , y_train)
knn.fit(X_train , y_train)
#%%
print("Logistic Regression Matrics")
print(f"Accuracy : {accuracy(y_test , lr.predict(X_test))} ")
print(f"Precision : {precision(y_test , lr.predict(X_test))} ")
print(f"Recall : {recall(y_test , lr.predict(X_test))} ")
print(f"F1 : {F1(y_test , lr.predict(X_test))} ")
print("==============")
print("KnnMatrics")
print(f"Accuracy : {accuracy(y_test , knn.predict(X_test))} ")
print(f"Precision : {precision(y_test , knn.predict(X_test))} ")
print(f"Recall : {recall(y_test , knn.predict(X_test))} ")
print(f"F1 : {F1(y_test , knn.predict(X_test))} ")


#%%
fpr , tpr , _ = roc_curve(y_test , lr.predict_proba(X_test)[: ,1])
plt.plot(fpr , tpr)

fpr , tpr , _ = roc_curve(y_test , knn.predict_proba(X_test)[: ,1])
plt.plot(fpr , tpr)
x = np.linspace(0 , 1 , 1000000)
plt.plot(x , x , linestyle = '--')

plt.title("ROC Curve")
plt.xlabel('False Postive Rate')
plt.ylabel('True Postive Rate')
plt.legend(['Logistic Regression , 10-NN'])

