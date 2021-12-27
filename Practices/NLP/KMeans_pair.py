#%%
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
points = np.random.rand(100,2)
#%%
print(points)
#%%
def distance(A, B):
    squares = [(a - b) ** 2 for a, b in zip(A, B)]
    return sum(squares) ** 0.5
#%%
def get_random_points(clusters):
    cluster_random_point = []
    for x in range(clusters):
        random =  np.random.randint(100, size=1)
        cluster_random_point.append(points[random])
    return cluster_random_point
get_random_points(3)


#%%
def upadteCentroids(points,centroids,clusters):
    for i in range(len(centroids)):
        centroids[i] = np.mean(points[i  == clusters] , axis=0)


def assignPoints(points , centroids , clusters):
    chage = False
    for i , p in enumerate(points):
        d , j = min([(distance(c , p ) , loc ) for loc , c in enumerate(centroids)])
        if j != clusters[i]:
            clusters[i]=j
            chage = True
        return chage



def KMeans(points, k=3):
    clusters = np.zeros(len(points))
    centroids = deepcopy(points[:k])
    while assignPoints(points, centroids, clusters):
        upadteCentroids(points, centroids, clusters)
    plt.scatter([p[0] for p in points], [p[1] for p in points], c=clusters)
    plt.show()

KMeans(3)