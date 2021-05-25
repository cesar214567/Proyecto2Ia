from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import numpy as np
from main import read_data



X = read_data()
#nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
#distances, indices = nbrs.kneighbors(X)


kdt = KDTree(X, leaf_size=30, metric='euclidean')
response = kdt.query(X, k=2, return_distance=False)

print(response)
