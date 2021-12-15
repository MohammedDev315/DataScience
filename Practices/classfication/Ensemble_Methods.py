#%%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

from mlxtend.classifier import StackingClassifier # <-- note: this is not from sklearn!

sns.set_style("whitegrid")


#%%
df = pd.read_csv('data/dataframe.csv')
X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1),
                                                    df.label,
                                                    random_state=123
                                                   )
sns.scatterplot(x='column1', y='column2', data=df, hue='label',
           palette='colorblind', alpha=.2)
plt.show()
#%%
model_names = ["lr_model", "knn_model", "rf_model", "et_model"]

for model_name in model_names:
    with open(f"models/{model_name}.pickle", "rb") as pfile:
        exec(f"{model_name} = pickle.load(pfile)")

model_vars = [eval(n) for n in model_names]
model_list = list(zip(model_names, model_vars))
for model_name in model_names:
    curr_model = eval(model_name)
    print(f'{model_name} score: {curr_model.score(X_test, y_test)}')

#%%
# create voting classifier
voting_classifer = VotingClassifier(estimators=model_list , voting='hard' , n_jobs=-1)
voting_classifer.fit(X_train , y_train)
y_pred = voting_classifer.predict(X_test)
print(accuracy_score(y_test, y_pred))
#%%
voting_classifer = VotingClassifier(estimators=model_list , voting="soft" , n_jobs=-1)
voting_classifer.fit(X_train , y_train)
y_pred = voting_classifer.predict(X_test)
print(accuracy_score(y_test, y_pred))
#%%
stacked = StackingClassifier(classifiers=model_vars , meta_classifier=LogisticRegression(), use_probas=False)
stacked.fit(X_train, y_train)
y_pred = stacked.predict(X_test)
print(accuracy_score(y_test , y_pred))
#%%



