#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score

iris_dataset = datasets.load_iris()
X_train , X_test , y_train , y_test = train_test_split( iris_dataset['data'], iris_dataset['target'], \
                                                            test_size=.3, random_state=42)
Knn = KNeighborsClassifier(n_neighbors=5)
Knn.fit(X_train,y_train)
print("Knn :-")
print(Knn.score(X_train,y_train))
print(Knn.score(X_test,y_test))


logit = LogisticRegression(C=5)
logit.fit(X_train,y_train)
print("Logistic")
print(logit.score(X_train,y_train))
print(logit.score(X_test,y_test))

#%%
logit.predict_proba(X_test[:5 , ])
#%%
print(X_test)
print(Knn.predict(X_test))
print(confusion_matrix(y_test , Knn.predict(X_test)))
print(confusion_matrix(y_test , logit.predict(X_test)))
#%%
knn_confusion = confusion_matrix(y_test, Knn.predict(X_test))
plt.figure(dpi=150)
sns.heatmap(knn_confusion, cmap=plt.cm.Blues, annot=True, square=True,
           xticklabels=iris_dataset['target_names'],
           yticklabels=iris_dataset['target_names'])
plt.xlabel('Predicted species')
plt.ylabel('Actual species')
plt.title('kNN confusion matrix')
plt.show()
