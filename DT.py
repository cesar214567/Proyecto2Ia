from sklearn.tree import DecisionTreeClassifier
from utilities import getY
from utilities import confuse_matrix
import numpy as np


def DT(X_train,X_test,X_train_data,X_test_data,option,method=None,show=False):
    y_train = getY(X_train)
    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=option)
    decision_tree = decision_tree.fit(X_train_data, y_train)
    y_test = getY(X_test)
    
    if show == True:
        result = decision_tree.predict(X_test_data)
        confuse_matrix(y_test,result, "DT_"+method)

    return decision_tree.score(X_test_data, y_test)
