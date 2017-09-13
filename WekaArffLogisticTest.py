import numpy as np, arff

dataset = arff.load(open('/media/mydrive/Weka_Sample_Datasets/datasets-UCI/UCI/iris.arff', 'r'))
data = np.array(dataset['data'])
train = data[:,0:4].astype(np.float)
labs = data[:,4]

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(train,labs)
