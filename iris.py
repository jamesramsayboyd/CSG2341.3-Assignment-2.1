from matplotlib import pyplot as plt
from sklearn.datasets import load_digits, load_iris
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
iris = load_iris()
X = iris.data
y = iris.target

### PRE-PROCESSING ###
# Iris dataset data points are text, so the sklearn LabelEncoder function is used to convert these to
# numerical data that the k-NN algorithm can handle with a Euclidean Distance metric
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training set (80%) and test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

### OPTIMISING PARAMETERS ###
accuracy_results = []
ks = range(1, 50)
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
# k value of around 10 appears optimal
knn = kNearestNeighbour(k=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

### DISPLAYING RESULTS ###
# Confusion Matrix
c_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9,6))
sb.heatmap(c_matrix, annot=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

### USING SKLEARN'S ACCURACY SCORE TOOL TO DISPLAY ALGORITHM PREDICTION ACCURACY ###
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
