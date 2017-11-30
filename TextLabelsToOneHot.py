import numpy as np, arff
from sklearn.preprocessing import LabelBinarizer

dataset = arff.load(open('iris.arff', 'r'))
data = np.array(dataset['data'])
text_labels = data[:,4]

print("#######################")
print("#One-Hot Encoding Demo#")
print("#######################")

# Print original labels
print("Text labels: %s\n" % text_labels)

# Encode labels
encoder = LabelBinarizer()
encoded_labels = encoder.fit_transform(text_labels)

# Print one-hot labels
print("One-Hot labels: %s\n" % encoded_labels)