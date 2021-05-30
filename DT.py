from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.tree import plot_tree
from SVM import getY
from main import get_partitions
from main import bootstrap
from main import confuse_matrix
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy as np


def DT(X_train,X_test,X_train_data,X_test_data,show=False):
    y_train = getY(X_train)
    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=10)
    decision_tree = decision_tree.fit(X_train_data, y_train)
    y_test = getY(X_test)
    
    if show == True:
        result = decision_tree.predict(X_test_data)
        confuse_matrix(y_test,result)

    return decision_tree.score(X_test_data, y_test)

#get_partitions(5,DT)
rs = RandomState(MT19937(SeedSequence(123456789)))
np.random.seed(3)
bootstrap(5,DT)