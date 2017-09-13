import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

kf = KFold(n_splits=10,shuffle=True)

for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    print(X_train)
    print("\n----------------------------------------")
