from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
#from KNN import save_data
from main import get_partitions
def getY(X):
    y = []
    for i in X:
        y.append(i['type'])
    return y

#Pipeline(steps=[('standardscaler', StandardScaler()),
#               ('linearsvc', LinearSVC(random_state=0, tol=1e-05))])


def SVM(X_train,X_test,X_train_data,X_test_data): 
    y_train = getY(X_train)
    #X_train_data, y_train = make_classification(n_features=len(X_train_data[0]), random_state=0)
    clf = make_pipeline(StandardScaler(),
                        LinearSVC(random_state=0, tol=1e-5))
    clf.fit(X_train_data, y_train)
    
    y_test = getY(X_test)
    j = 0 
    goods = 0
    bads =0
    for i in X_test_data:
        res = clf.predict([i])
        if res == y_test[j]:
            print('Good')
            goods+=1
        else:
            bads+=1
            print("Bad")
        j += 1
        
get_partitions(7,SVM)