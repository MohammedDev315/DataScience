#%%
from sklearn.cluster import DBSCAN, SpectralClustering, MeanShift
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use("seaborn")

#%%
def get_data(k,num_points=100):
    np.random.seed(9)
    data = []
    for i in range(0,k):
        for _ in range(0,num_points):
            data.append([np.random.normal(6*i),np.random.normal(i)])
    x1,y1 = zip(*data)
    return data
#%%
data = get_data(6)
x,y = zip(*data)
plt.scatter(x, y, s=20)
plt.show()
#%%
# Standardize our data for DBSCAN and fit DBSCAN
X = StandardScaler().fit_transform(data)
db = DBSCAN(eps=0.15, min_samples=3).fit(X)

# Let's find the observations DBSCAN called "core"
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

unique_labels = set(labels)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
plt.figure(dpi=200)
show_core = True
show_non_core = True
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)
    if show_core:
        xy = X[class_member_mask & core_samples_mask]
        x, y = xy[:, 0], xy[:, 1]
        plt.scatter(x, y, c=col, edgecolors='k', s=20, linewidths=1.1)  # add black border for core points

    if show_non_core:
        xy = X[class_member_mask & ~core_samples_mask]
        x, y = xy[:, 0], xy[:, 1]
        plt.scatter(x, y, c=col, s=20, linewidths=1.1)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


#%%
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

plt.figure(dpi=200)
centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)
x,y = zip(*X)
plt.scatter(x,y)
plt.show()

# estimate bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

#%%
from itertools import cycle
plt.figure(1)
plt.clf()
plt.figure(dpi=200)
colors = cycle('bcmgkr')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1] ,  col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'X', markerfacecolor='k',
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

#%%
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
ypred = km.fit_predict(X)
x,y = zip(*X)
plt.figure(dpi=200)
plt.scatter(X[:,0],X[:,1],c=plt.cm.rainbow(ypred*20),s=14)
plt.show()

#%%
from time import time
from sklearn.cluster import AgglomerativeClustering

for linkage in ('ward' , 'average' , 'complete' ):
    clustering = AgglomerativeClustering(linkage = linkage , n_clusters=3)
    t0 = time()
    clustering.fit(X)
    print("%s : %.2fs" % (linkage, time() - t0))
    x  , y = zip(*X)
    plt.figure(dpi=200)
    plt.scatter(x, y, c=plt.cm.rainbow(clustering.labels_ * 20), s=14)
    plt.title("Linkage Type: %s" % linkage)
    plt.show()

#%%
from sklearn.cluster import SpectralClustering
sc = SpectralClustering(n_clusters=3)
ypred  = sc.fit_predict(X)
x , y =  zip(*X)
plt.figure(dpi=200)
plt.scatter(X[:,0],X[:,1],c=plt.cm.rainbow(ypred*20),s=14)
plt.show()

#%%
from mpl_toolkits.mplot3d import Axes3D
def get_high_dim_data(k,num_dim=20,num_points=100):
    np.random.seed(9)
    data = []
    modifiers = []
    for i in range(0,k):
        modifiers = np.random.randint(k,size=num_dim)
        for _ in range(0,num_points):
            data_vals = []
            for j in range(num_dim):
                data_vals.append(np.random.normal(modifiers[j]*i))
            data.append(data_vals)
    return data

data = get_high_dim_data(10,num_dim=3)
x,y,z = zip(*data)
plt.style.use("seaborn-poster")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)

from sklearn.manifold import TSNE
num_points = 100
num_clust = 10
num_dim = 30
high_data = get_high_dim_data(num_clust, num_points=num_points, num_dim=num_dim)
model = TSNE(n_components=2, random_state=0,verbose=2)
low_data = model.fit_transform(high_data)


colorize = []
for i in range(num_clust):
    for _ in range(num_points):
        colorize.append(plt.cm.rainbow(i*20))
x,y = zip(*low_data)
plt.scatter(x,y,c=colorize,s=40)
plt.show()


#%%
from sklearn import datasets
from sklearn.manifold import TSNE
digits = datasets.load_digits()
X = digits.data
model = TSNE(n_components=2, random_state=0,verbose=0)
low_data = model.fit_transform(X)
target = digits.target
target_names = digits.target_names


colors = cycle(['r','g','b','c','m','y','orange','k','aqua','yellow'])
target_ids = range(len(target_names))
plt.figure(dpi=150)
for i, c, label in zip(target_ids, colors, target_names):
    plt.scatter(low_data[target == i, 0], low_data[target == i, 1], c=c, label=label, s=15, alpha=1)
plt.legend(fontsize=10, loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#333333')
plt.xlim(-100,100);
plt.title("Digit Clusters with TSNE", fontsize=12)
plt.ylabel("Junk TSNE Axis 2", fontsize=12)
plt.xlabel("Junk TSNE Axis 1", fontsize=12);
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
