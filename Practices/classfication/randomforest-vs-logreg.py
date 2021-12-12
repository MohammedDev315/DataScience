#%%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.cm as cm
from sklearn.neighbors import KNeighborsClassifier


X = np.random.rand(100, 2)
y1 = [1 if i[0] > i[1] else 0 for i in X]


def visualize(X, y, bdry='diag'):
    c = cm.rainbow(np.linspace(0, 1, 2))
    plt.scatter([i[0] for i in X], [i[1] for i in X], color=[c[i] for i in y], alpha=.5)
    # Plot the true decision boundary
    if bdry == 'diag':
        plt.plot([0, 1], [0, 1], 'k--')
    elif bdry == 'quadrant':
        plt.plot([0, 1], [0.5, 0.5], 'k--')
        plt.plot([0.5, 0.5], [0, 1], 'k--')
    plt.grid(True)
visualize(X, y1)
# plt.show()


def quick_test(model , X , y):
    xtrain, xtest , ytrain, ytest = train_test_split(X , y , test_size=0.3)
    model.fit(xtrain , ytrain)
    return model.score(xtest , ytest)
def quick_test_afew_times(model , X , y , n=10):
    return  np.mean([quick_test(model , X , y ) for j in range(n) ])

#%%
logreg = LogisticRegression(penalty='none')
print(f"Logreg : {quick_test_afew_times(logreg, X, y1)} ")

decisiontree = DecisionTreeClassifier(max_depth=4)
print(f"DecisionTree : {quick_test_afew_times(decisiontree, X, y1)} ")

randomforest = RandomForestClassifier(n_estimators=100)
print(f"RandomForest : {quick_test_afew_times(randomforest, X, y1)} ")

Knn = KNeighborsClassifier(n_neighbors=5)
print(f"Knn {quick_test_afew_times(Knn , X , y1)} ")

#%%
decisiontree.fit(X, y1)
grid = np.mgrid[0:1.02:0.02, 0:1.02:0.02].reshape(2,-1).T
visualize(grid, decisiontree.predict(grid))
plt.show()
#%%
logreg.fit(X, y1)
grid = np.mgrid[0:1.02:0.02, 0:1.02:0.02].reshape(2,-1).T
visualize(grid, logreg.predict(grid))
plt.show()
#%%
y2 = [1 if (0.5-i[0])*(0.5-i[1])>0 else 0 for i in X]
visualize(X, y2, bdry='quadrant')
logreg = LogisticRegression()
print(f"Logistic Regression with y2 :  {quick_test_afew_times(logreg , X , y2)} ")
decisiontree = DecisionTreeClassifier(max_depth=4)
print(f"Decisiontree with y2 : {quick_test_afew_times(decisiontree , X , y2)} ")
randomforest = RandomForestClassifier(n_estimators=100)
print(f"Randomforest wtih y2  : {quick_test_afew_times(randomforest , X ,y2)} ")
Knn = KNeighborsClassifier(n_neighbors=3)
print(f"Knn {quick_test_afew_times(Knn , X , y2)} ")
#%%
decisiontree.fit(X, y2)
grid = np.mgrid[0:1.02:0.02, 0:1.02:0.02].reshape(2,-1).T
visualize(grid, decisiontree.predict(grid), bdry='quadrant')
plt.show()
#%%
logreg.fit(X, y2)
grid = np.mgrid[0:1.02:0.02, 0:1.02:0.02].reshape(2,-1).T
visualize(grid, logreg.predict(grid), bdry='quadrant')
plt.show()
#%%


