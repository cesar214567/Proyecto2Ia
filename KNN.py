from sklearn.neighbors import NearestNeighbors
import numpy as np
from main import read_data



X = read_data()
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
