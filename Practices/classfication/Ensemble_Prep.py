#%%
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model, neighbors, ensemble

df = pd.read_csv('data/dataframe.csv')
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    df.drop('label', axis=1), df.label, random_state=123)

lr_model = linear_model.LogisticRegression(solver = "lbfgs" , random_state = 1)
knn_model = neighbors.KNeighborsClassifier()
rf_model = ensemble.RandomForestClassifier(n_estimators=100 , random_state=1)
et_model = ensemble.ExtraTreesClassifier(n_estimators=100 , random_state=1)

models = ["lr_model", "knn_model", "rf_model", "et_model"]


#%%
import pickle
for model_name in models:
    curr_model = eval(model_name)
    curr_model.fit(X_train, y_train)
    with open(f"models/{model_name}.pickle", "wb") as pfile:
        pickle.dump(curr_model, pfile)
#%%

