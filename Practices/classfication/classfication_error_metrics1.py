#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score , log_loss
from sklearn.metrics import confusion_matrix
# from ipywidgets import interactive, FloatSlider
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.dummy import DummyClassifier


#%%
plt.style.use('ggplot')
np.set_printoptions(suppress=True)
iris_dataset = datasets.load_iris()
iris_df = pd.DataFrame(iris_dataset['data'])
print(iris_df.columns)

# sns.pairplot(iris_dataset, hue='target')
# plt.show()

X_train, X_test, label_train, label_test = train_test_split(iris_dataset['data'], iris_dataset['target'], \
                                                            test_size=0.3, random_state=41)
#%%
Knn = KNeighborsClassifier(n_neighbors=5)
Knn.fit(X_train , label_train)
print("KNN")
print(f"Trining : {100*Knn.score(X_train , label_train)}")
print(f"Trining : {100*Knn.score(X_test , label_test)}")

logit = LogisticRegression(C=0.95)
logit.fit(X_train , label_train)
print("LogisticRegressin")
print(f"{100 * logit.score(X_train , label_train)}")
print(f"{100 * logit.score(X_test , label_test)}")
#%%
print(logit.predict_proba(X_train[:5 , ]))
#%%
print("NN confusion matrix")
print( confusion_matrix(label_test , Knn.predict(X_test)))
print("Confusion matrix for silly model where we predict all 2's: \n\n", \
      confusion_matrix(label_test, [2]*len(label_test)))
#%%
knn_confusion = confusion_matrix(label_test, Knn.predict(X_test))
plt.figure(dpi=150)
sns.heatmap(knn_confusion, cmap=plt.cm.Blues, annot=True, square=True,
           xticklabels=iris_dataset['target_names'],
           yticklabels=iris_dataset['target_names'])
plt.xlabel('Predicted species')
plt.ylabel('Actual species')
plt.title('kNN confusion matrix')
plt.show()
#%%
logit_confusion = confusion_matrix(label_test, logit.predict(X_test))
plt.figure(dpi=150)
sns.heatmap(logit_confusion, cmap=plt.cm.Blues, annot=True, square=True,
           xticklabels=iris_dataset['target_names'],
           yticklabels=iris_dataset['target_names'])

plt.xlabel('Predicted species')
plt.ylabel('Actual species')
plt.title('Logistic regression confusion matrix');
plt.savefig("confusion_matrix_logit_iris")
plt.show()
#%%
# Let's read in some credit card data!
df = pd.read_csv('creditcard.csv')
print(df.head())
print(df.Class.value_counts())
#%%
X_train, X_test , y_train , y_test = train_test_split(df.iloc[: , 1 : -1] , df.iloc[: , -1] , random_state = 42)
lm = LogisticRegression(C =100 )
lm.fit(X_train , y_train)
print(f"Logistic score :  {lm.score(X_train , y_train) } ")
#%%
def make_confusion_matrix(model, threshold=0.5):
    # Predict class 1 if probability of being in class 1 is greater than threshold
    # (model.predict(X_test) does this automatically with a threshold of 0.5)
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    fraud_confusion = confusion_matrix(y_test, y_predict)
    plt.figure(dpi=80)
    sns.heatmap(fraud_confusion, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
           xticklabels=['legit', 'fraud'],
           yticklabels=['legit', 'fraud']);
    plt.xlabel('prediction')
    plt.ylabel('actual')
make_confusion_matrix(lm , 0.2 )
plt.show()
#%%
y_predict = lm.predict(X_test)
print("Default Threshold")
print(precision_score(y_test , y_predict))
print(recall_score(y_test , y_predict))
#%%
y_predict = (lm.predict_proba(X_test)[:,1] > 0.04)
print(y_predict.head())
print(precision_score(y_test , y_predict))
print(recall_score(y_test , y_predict))
#%%
precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_test, lm.predict_proba(X_test)[:,1] )
plt.figure(dpi=80)
plt.plot(threshold_curve, precision_curve[1:],label='precision')
plt.plot(threshold_curve, recall_curve[1:], label='recall')
plt.legend(loc='lower left')
plt.xlabel('Threshold (above this probability, label as fraud)');
plt.title('Precision and Recall Curves')
plt.show()
#%%
# Or we can just ask sklearn
y_predict = lm.predict(X_test)
print(f1_score(y_test , y_predict))
#%%
y_predict = (lm.predict_proba(X_test)[: , 1] > 0.06)
lm.predict(X_test)
print(f1_score(y_test , y_predict))
#%%
print(fbeta_score(y_test, y_predict, 2))
#%%
fpr, tpr, thresholds = roc_curve(y_test, lm.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr,lw=2)
plt.plot([0,1],[0,1],c='violet',ls='--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve for fraud problem')
print("ROC AUC score = ", roc_auc_score(y_test, lm.predict_proba(X_test)[:,1]))
plt.show()
#%%
print(f"log_loss : log_loss(y_test, lm.predict_proba(X_test))")
dc = DummyClassifier()
dc.fit(X_train, y_train)
print(log_loss(y_test, dc.predict_proba(X_test)))
print(lm.score(X_test, y_test))
print(dc.score(X_test, y_test))