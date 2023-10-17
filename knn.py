import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Helper functions
def most_common(lst):
    '''Returns the most common element in a list'''
    return max(set(lst), key = lst.count)

def euclidean(point, data):
    '''Euclidean distance between a point & data'''
    return np.sqrt(np.sum((point - data)**2, axis=1))


#Constructing class
class KNeighborsClassifier():
    def __init__(self, k=5, dist_metric = euclidean):
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])

        return list(map(most_common, neighbors))

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy

class KNeighborsRegressor:
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])

        return np.mean(neighbors, axis=1)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        ssre = sum((y_pred - y_test)**2)
        return ssre

# from ucimlrepo import fetch_ucirepo
#
# # fetch dataset
# iris = fetch_ucirepo(id=53)
#
# # data (as pandas dataframes)
# X = iris.data.features
# y = iris.data.targets
#
# # metadata
# print(iris.metadata)
#
# # variable information
# print(iris.variables)

# Iris Dataset for Classifier Implementation
#unpack iris dataset
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#pre-process data
ss = StandardScaler().fit(X_train)
X_train, X_test = ss.transform(X_train), ss.transform(X_test)

#Test knn model across varying ks
accuracies = []
ks = range(1, 30)
for k in ks:
    knn = KNeighborsClassifier(k = k)
    knn.fit(X_train, y_train)
    accuracy = knn.evaluate(X_test, y_test)
    accuracies.append(accuracy)

# Visualise accuracy vs. k
fig, ax = plt.subplots()
ax.plot(ks, accuracies)
ax.set(xlabel="k",
       ylabel="Accuracy",
       title="Performance of KNN Classifier")
#plt.show()

#Housing Dataset for Regressor Implementation
# Unpack the California housing dataset, from StatLib repository
housing = datasets.fetch_california_housing()
X = housing['data'][:500]
y = housing['target'][:500]

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocess data
ss = StandardScaler().fit(X_train)
X_train, X_test = ss.transform(X_train), ss.transform(X_test)

# Test knn model across varying ks
accuracies = []
ks = range(1, 30)
for k in ks:
    knn = KNeighborsRegressor(k=k)
    knn.fit(X_train, y_train)
    accuracy = knn.evaluate(X_test, y_test)
    accuracies.append(accuracy)

# Visualize accuracy vs. k
fig, ax = plt.subplots()
ax.plot(ks, accuracies)
ax.set(xlabel="k",
       ylabel="SSRE",
       title="Performance of KNN Regressor")
plt.show()
