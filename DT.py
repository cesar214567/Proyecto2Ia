from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.tree import plot_tree
from SVM import getY
from main import get_partitions

def DT(X_train,X_test,X_train_data,X_test_data):
    #iris = load_iris()
    y_train = getY(X_train)
    #print(y_train)
    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
    decision_tree = decision_tree.fit(X_train_data, y_train)
    print(plot_tree(decision_tree)) 
    #r = export_text(decision_tree, feature_names=iris['feature_names'])
    
    y_test = getY(X_test)
    j = 0 
    for i in X_test_data:
        res = decision_tree.predict([i])
        if res == y_test[j]:
            print('Good')
        else:
            print("Bad")
        j += 1

get_partitions(5,DT)