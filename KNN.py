from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import numpy as np
from main import get_partitions
from main import save_data


#X = read_data()
#nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
#distances, indices = nbrs.kneighbors(X)

def find_type(diccionary):
    mx = 0
    mv = ''
    for t, value in diccionary.items():
        if mx < value:
            mv = t
            mx = value
    return mv
      


def results(X_train,X_test,X_train_data,X_test_data,results_data):
    i = 0
    goods =0
    bads = 0
    for results in results_data:
        diccionary = {"fear" : 0, "anger": 0, "contempt": 0, "happy": 0, "disgust": 0,"sadness": 0, "surprise": 0}
        for result in results:
            diccionary[X_train[result]['type']] += 1
        
        if X_test[i]['type'] == find_type(diccionary):
            #print('Good')
            goods+=1
        else:
           # print('Bad')
            bads+=1
            #print(X_test[i]['type'])
            #print(find_type(diccionary))
        i+=1
    return goods/(goods+bads)
                

def knn(X_train,X_test,X_train_data,X_test_data, show = False):

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X_train_data)
    #kdt = KDTree(X_train_data, leaf_size=30, metric='euclidean')
    #response = kdt.query(X_test_data, k=5, return_distance=False)
    distances, indices = nbrs.kneighbors(X_test_data)
    return(results(X_train,X_test,X_train_data,X_test_data,indices))


    
save_data()
get_partitions(7,knn)
