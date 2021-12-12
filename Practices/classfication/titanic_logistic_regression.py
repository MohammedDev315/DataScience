#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve,  precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv("train.csv")
df_none = train_df.isnull().sum()
total_count = train_df["Survived"].value_counts()
print(total_count)
sns.heatmap( train_df.isnull() ,  yticklabels = False , cbar = False , cmap = "viridis" )
plt.show()
sns.countplot(x = 'Survived' , hue='Sex' ,  data = train_df)
plt.show()
#%%
#this digram can be helpfult to understand the distribution of age
#amount different class, it is clear to see, the walthy class 1 people
# are older than other class, so, we can use this info to fill the null
# values of age base on class groups
sns.boxplot(x="Pclass" , y="Age" , data = train_df)
plt.show()
#%%
print(train_df.loc[: , 'Pclass' ].value_counts())
#%%
#Base line
train_df = train_df.loc[train_df.loc[:, "Age"].notna()]
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
train_df_numeric = train_df.select_dtypes(include=numerics)
x = train_df_numeric.drop( "Survived", axis = 1)
y = train_df_numeric["Survived"]
X_train, X_test, y_train, y_test = train_test_split( x , y ,test_size = 0.2, random_state=20)
lm1 = LogisticRegression(C=1000)
lm1.fit( X_train , y_train)
scores = cross_val_score(lm1, X_train, y_train, cv=5)
print(scores.mean())
#%%
print(confusion_matrix(y_test , lm1.predict(X_test)))
print(classification_report(y_test , lm1.predict(X_test)))
#%%
train_df = train_df.loc[train_df.loc[:, "Age"].notna()]
train_df = train_df.loc[train_df.loc[:, "Embarked"].notna()]
Em = pd.get_dummies(train_df["Embarked"] ,  drop_first=True)
Sex = pd.get_dummies(train_df["Sex"] ,  drop_first=True )
train_df = train_df.join(Em, lsuffix="_l", rsuffix="_r")
train_df = train_df.join(Sex, lsuffix="_l", rsuffix="_r")
# train_df = pd.concat([train_df , Sex , Em ] , axis=1 )
#%%
def get_title(data_in):
    title = data_in.split(",")[1].split()[0].replace("." , "")
    if title not in ["Mr" , "Miss" , "Mrs" , "Master" , "Rev" , "Dr"]:
        title = "Other"
    return title
train_df["Title"] = train_df.loc[: , "Name" ].apply(get_title)
# print(train_df.loc[: , "Title"].value_counts())
#%%
#Get dummies for title
title_dummy = pd.get_dummies(train_df.loc[: , "Title"] , drop_first=True)
train_df = train_df.join(title_dummy)
#%%
x = train_df.drop( ["PassengerId" , 'Embarked' ,  "Name" , "Sex" , "Ticket" ,  "Survived" , "Cabin"], axis = 1)
y = train_df["Survived"]
X_train, X_test, y_train, y_test = train_test_split( x , y ,test_size = 0.2, random_state=20)
#%%
sns.pairplot(train_df , hue="Survived")
plt.show()
#%%
print(X_train.columns)
#%%
col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
sns.boxplot(x='Survived', y=col[4], data=train_df)
plt.show()
#%%
lm2 = LogisticRegression(C=1000)
lm2.fit( X_train , y_train)
scores = cross_val_score(lm2, X_train, y_train, cv=5)
print(scores.mean())
#%%
#Scalling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
#%%
lm3 = LogisticRegression()
lm3.fit(X_train_scaled , y_train)
score3_train = cross_val_score( lm3 , X_train_scaled , y_train ,cv=5)
score3_test = lm3.score(X_test_scaled , y_test)
print(score3_train.mean())
print(score3_test)
#%%
# Change C using GridSearchCV
parm = {"C":np.logspace(-20,20,1000), "penalty":["l1","l2"]}# l1 lasso l2 ridge
grid = GridSearchCV(lm3, param_grid = parm, cv=5)
grid.fit(X_train_scaled, y_train)
print(grid.best_params_)
#%%
lm4 = LogisticRegression(C=0.288 , penalty="l2")
lm4.fit(X_train_scaled , y_train)
score4_train = cross_val_score( lm4 , X_train_scaled , y_train ,cv=5)
score4_test = lm4.score(X_test_scaled , y_test)
print(score4_train.mean())
print(score4_test)
print(confusion_matrix(y_test , lm4.predict(X_test)))
print(classification_report(y_test , lm4.predict(X_test)))
#%%
y_test_new_threshold = (lm4.predict_proba(X_test)[: , 1 ] >= 0.9)
print(confusion_matrix(y_test , y_test_new_threshold))
# y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
# fraud_confusion = confusion_matrix(y_test, y_predict)
#%%
precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_test, lm4.predict_proba(X_test)[:,1] )
plt.figure(dpi=80)
plt.plot(threshold_curve, precision_curve[1:],label='precision')
plt.plot(threshold_curve, recall_curve[1:], label='recall')
plt.legend(loc='lower left')
plt.xlabel('Threshold')
plt.title('Precision')
plt.show()



