from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from utilities import getY
from utilities import confuse_matrix

def find_type(diccionary):
    mx = 0
    mv = ''
    for t, value in diccionary.items():
        if mx < value:
            mv = t
            mx = value
    return mv
      
def results(X_train,X_test,X_train_data,X_test_data,results_data,method,option):
    i = 0
    goods =0
    bads = 0
    final_results = []
    res = getY(X_test)
    for results in results_data:
        diccionary = {"fear" : 0, "anger": 0, "contempt": 0, "happy": 0, "disgust": 0,"sadness": 0, "surprise": 0}
        for result in results:
            diccionary[X_train[result]['type']] += 1
        
        final_results.append(find_type(diccionary))


        if X_test[i]['type'] == find_type(diccionary):
            goods+=1
        else:
            bads+=1
        i+=1
    confuse_matrix(res,final_results,'KNN_'+ str(option)+ '_' + str(method))
    return goods/(goods+bads)
                

def knn(X_train,X_test,X_train_data,X_test_data,option,method=None,show = False):
    nbrs = NearestNeighbors(n_neighbors=option, algorithm='ball_tree').fit(X_train_data)
    distances, indices = nbrs.kneighbors(X_test_data)
    return(results(X_train,X_test,X_train_data,X_test_data,indices,method,option))
