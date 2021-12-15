#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/clean_data.csv")
print(df.columns)
#%%
df_numeric = df.select_dtypes(include=np.number)
x = df.drop("LOAN_DEFAULT" , axis = 1)
y = df["LOAN_DEFAULT"]
x_numeric = df_numeric.drop("LOAN_DEFAULT" , axis = 1)
y_numeric = df_numeric["LOAN_DEFAULT"]

X_train , X_vald , y_train , y_vald = train_test_split(x , y , test_size = 0.2, random_state=42)


#%%
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train , y_train)
print(f"Knn Train{model.score(X_train , y_train)}")
print(f"Knn Val{model.score(X_vald , y_vald)}")
y_pred = model.predict(X_vald)
cf_matrix = confusion_matrix(y_vald, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True)
plt.show()

#%%
model = LogisticRegression()
model.fit(X_train , y_train)
print(f"Logistic Train{model.score(X_train , y_train)}")
print(f"Logistic Val{model.score(X_vald , y_vald)}")
y_pred = model.predict(X_vald)
cf_matrix = confusion_matrix(y_vald, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True)
plt.show()



