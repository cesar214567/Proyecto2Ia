from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
#from KNN import save_data
from main import get_partitions
from main import bootstrap
def getY(X):
    y = []
    for i in X:
        y.append(i['type'])
    return y

def SVM(X_train,X_test,X_train_data,X_test_data, show=False): 
    y_train = getY(X_train)
    clf = make_pipeline(StandardScaler(),
                        LinearSVC(random_state=0, tol=1e-5))
    clf.fit(X_train_data, y_train)
    
    y_test = getY(X_test)
    return clf.score(X_test_data,y_test)
        
get_partitions(7,SVM)
bootstrap(7, SVM)
