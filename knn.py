from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sb

### UTILITY FUNCTIONS ###
# Finds the most common class in a list of nearest neighbour data points
def most_common(neighbour_list):
    return max(set(neighbour_list), key=neighbour_list.count)


# Finds the Euclidean distance between two data points
def euclidean(target_datapoint, neighbour_datapoint):
    return np.sqrt(np.sum((target_datapoint - neighbour_datapoint)**2, axis=1))


### CLASS ###
class kNearestNeighbour:
    def __init__(self, k=5):
        self.k = k
        self.X_train = y_train
        self.y_train = X_train

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        nearest_neighbours = []
        for i in X_test:
            euclidean_distances = euclidean(i, self.X_train)
            y_sorted = [y for _, y in sorted(zip(euclidean_distances, self.y_train))]
            nearest_neighbours.append(y_sorted[:self.k])
        return list(map(most_common, nearest_neighbours))

    def accuracy(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy


### LOAD DATA ###
digits = load_digits()
X = digits.images
y = digits.target

# Plot the first few images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.ravel()
for i in np.arange(0, 10):
    axes[i].imshow(X[i], cmap='gray')
    axes[i].set_title("Digit: %s" % y[i])
    axes[i].axis('off')
plt.subplots_adjust(wspace=0.5)
plt.show()

### PRE-PROCESSING ###
# X is a 3D array, i.e. 1797 images of 8x8 pixels
# Convert to 2D array, i.e. 1797 x 64 element arrays
X = digits.images.reshape((len(digits.images), -1))

# Split the data into training set (80%) and test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

### OPTIMISING PARAMETERS ###
accuracy_results = []
ks = range(1, 20)
for k in ks:
    knn = kNearestNeighbour(k=k)
    knn.fit(X_train, y_train)
    accuracy = knn.accuracy(X_test, y_test)
    accuracy_results.append(accuracy)

fig, ax = plt.subplots()
ax.plot(ks, accuracy_results)
ax.set(xlabel="k",
       ylabel="Accuracy",
       title="Accuracy of k-NN algorithm with varying values of k")
plt.show()

### RUNNING ALGORITHM WITH OPTIMISED PARAMETERS ###
knn = kNearestNeighbour(k=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

### DISPLAYING RESULTS ###
# Classification report
print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

# Confusion Matrix
c_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9,6))
sb.heatmap(c_matrix, annot=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()


