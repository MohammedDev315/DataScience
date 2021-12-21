#%%
# import numpy as np
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
from matplotlib import pyplot as plt
#%%
digits = datasets.load_digits()
print(digits.data.shape)
#%%
print(list(digits.data))
#%%
digits.data[166].reshape(8,8)
#%%
print(digits.data.shape)
#%%
X_centered = digits.data - digits.data.mean()
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X_centered, y, test_size=0.5,random_state=42)
print(X_train.shape)
#%%
pca = PCA(n_components=2)
pca.fit(X_train)
pcafeatures_train = pca.transform(X_train)
#%%
from itertools import cycle

def plot_PCA_2D(data, target, target_names):
    colors = cycle(['r','g','b','c','m','y','orange','w','aqua','yellow'])
    target_ids = range(len(target_names))
    plt.figure(figsize=(10,10))
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(data[target == i, 0], data[target == i, 1],
                   c=c, label=label, edgecolors='gray')
    plt.legend()
    plt.show()
#%%
# plot of all the numbers
plot_PCA_2D(pcafeatures_train, target=y_train, target_names=digits.target_names)
#%%
X_transf = pca.transform(X_train)
print("shape of original X_train:", X_train.shape)
print("shape of X_train using 2 principal components:", X_transf.shape, "\n")
print(X_transf)
#%%
print(pca.explained_variance_ratio_)
#%%
pd.DataFrame(pca.components_, index = ['PC1','PC2'])
#%%
pca2 = PCA(n_components=15)
pca2.fit(X_train)
pcafeatures_train2 = pca2.transform(X_train)
#%%
plt.plot(pca2.explained_variance_ratio_)
plt.xlabel('# components')
plt.ylabel('explained variance')
plt.title('Scree plot for digits dataset')
plt.show()
#%%
print(pca2.explained_variance_ratio_)
#%%
plt.plot(np.cumsum(pca2.explained_variance_ratio_))
plt.xlabel('# components')
plt.ylabel('cumulative explained variance');
plt.title('Cumulative explained variance by PCA for digits')
plt.show()
#%%









