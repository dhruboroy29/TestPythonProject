import numpy as np, pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

print("###################################")
print("#One-Hot Encoding Demo - iris.arff#")
print("###################################")

# Load dataset1
dataset1 = arff.loadarff('iris.arff')

# Extract data
df = pd.DataFrame(dataset1[0])
data1 = df.select_dtypes(exclude=[object]).as_matrix()

# Extract labels
lab1 = df.select_dtypes(include=[object])

# How many labels?
num_labels = lab1.size

# How many classes?
num_classes = np.unique(lab1).size

# Print original labels
print("Text labels: %s\n" % np.unique(lab1))

# Encode labels as numbers
encoder = LabelEncoder()
encoded_labels = lab1.apply(encoder.fit_transform)

# Convert to one-hot
onehot = OneHotEncoder()
onehot.fit(encoded_labels)
onehot_labels = onehot.transform(encoded_labels).toarray()

# Print one-hot labels
print("One-Hot labels: %s\n" % onehot_labels)

#--------------------------------------------------------------------------------------------------------------#

print("############################################")
print("#One-Hot Encoding Demo - radar1_scaled.arff#")
print("############################################")

# Load dataset2
dataset2 = arff.loadarff('radar1_scaled.arff')

# Extract data
df = pd.DataFrame(dataset2[0])
data2 = df.select_dtypes(exclude=[object]).as_matrix()

# Extract labels
lab2 = df.select_dtypes(include=[object])

# How many labels?
num_labels = lab2.size

# How many classes?
num_classes = np.unique(lab1).size

# Print original labels
print("Text labels: %s\n" % np.unique(lab2))

# Encode labels as numbers
encoder = LabelEncoder()
encoded_labels = lab2.apply(encoder.fit_transform)

# Convert to one-hot
onehot = OneHotEncoder()
onehot.fit(encoded_labels)
onehot_labels = onehot.transform(encoded_labels).toarray()

# Print one-hot labels
print("One-Hot labels: %s\n" % onehot_labels)