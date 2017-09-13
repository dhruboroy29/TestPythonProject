from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Read in Iris data
iris = load_iris()

# Create X (features) and y (response)
X = iris.data
y = iris.target

logreg = LogisticRegression()

acc_10fold = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')

print("10-fold CV accuracies: ", acc_10fold)
print("10-fold CV mean accuracy: ", acc_10fold.mean())