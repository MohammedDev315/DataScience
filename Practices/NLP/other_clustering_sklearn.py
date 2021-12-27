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





