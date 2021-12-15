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


df = pd.read_csv("data/train.csv")
#%%
df_numeric = df.select_dtypes(include=np.number)
x = df.drop("VOTERID_FLAG" , axis = 1)
y = df["VOTERID_FLAG"]

x_numeric = df_numeric.drop("VOTERID_FLAG" , axis = 1)
y_numeric = df_numeric["VOTERID_FLAG"]

X_train , X_vald , y_train , y_vald = train_test_split(x , y , test_size = 0.2, random_state=42)
X_train_numeric , X_vald_numeric , y_train_numeric , y_vald_numeric = train_test_split(x_numeric , y_numeric , test_size = 0.2, random_state=42)

#%%
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_vald_numeric , y_vald_numeric)
print(f"Knn Train{model.score(X_train_numeric , y_train_numeric)}")
print(f"Knn Val{model.score(X_vald_numeric , y_vald_numeric)}")

#%%
model = LogisticRegression()
model.fit(X_vald_numeric , y_vald_numeric)
print(f"Logistic Train{model.score(X_train_numeric , y_train_numeric)}")
print(f"Logistic Val{model.score(X_vald_numeric , y_vald_numeric)}")



