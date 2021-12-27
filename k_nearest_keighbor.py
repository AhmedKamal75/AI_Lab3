import statistics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import math
import dill
from Naive_Bayes import get_result

FILE_NAME_1 = "data/knn.txt"
FILE_NAME_2 = "data/knn.pickle"


def distance(point_1, point_2):
    return np.sqrt(np.sum((point_1 - point_2) ** 2))


class KNN:
    def __init__(self, k=11):
        self.k = k
        self.count = 0

    def fit(self, X, y):
        self.y = y
        self.X = X

    def predict(self, x):
        return np.array([self.predict_instance(instance_x) for instance_x in x])

    def predict_instance(self, x):
        self.count += 1
        distances = np.array([distance(x, x_train) for x_train in self.X])
        indices_of_sorted_distances = np.argsort(distances)[:self.k]
        nearest_labels = np.array([self.y[i] for i in indices_of_sorted_distances])
        label = float(statistics.mode(nearest_labels))
        print(f"{(self.count / self.X.shape[0]):.2%}")
        return label


def knn(X_train, y_train, X_test, y_test, write_to_file=False, read_of_file=False, text=False):
    results = {}
    if read_of_file:
        if text:
            with open(FILE_NAME_1, "rt") as file:
                for line in file:
                    data = line.rstrip().split(sep=",")
                    k, accuracy = float(data[0]), float(data[1])
                    results[accuracy] = k
        else:
            with open(FILE_NAME_2, "rb") as file:
                running = True
                try:
                    while running:
                        result, k = dill.load(file)
                        results[result.get("accuracy")] = (result, k)
                except EOFError:
                    running = False
    else:
        if write_to_file:
            if text:
                with open(FILE_NAME_1, "wt") as file:
                    pass
            else:
                with open(FILE_NAME_2, "wb") as file:
                    pass
        X = np.array(X_train)
        y = np.array(y_train)
        start = math.sqrt(X_train.shape[0]) / 5
        # delete thins
        start = start / 10
        #
        for k in np.arange(start, 10 * start, start):
            model = KNN(k=int(k))
            model.fit(X, y)
            predictions = model.predict(np.array(X_test))
            result = get_result(y_test, predictions)
            results[result.get("accuracy")] = [result, k]
            print(f"k: {k}, accuracy: {result.get('accuracy')}")
            if text:
                if write_to_file:
                    with open(FILE_NAME_1, "at") as file:
                        file.write(f"{int(k)},{result.get('accuracy')}\n")
            else:
                if write_to_file:
                    with open(FILE_NAME_2, "ab") as file:
                        dill.dump((result, k), file=file)

    answer = results.get(sorted(results)[0])
    print(answer)
    return results.get(sorted(results)[0])


def knn_(X_train, y_train, X_test, y_test):
    X = np.array(X_train)
    y = np.array(y_train)
    Xt = np.array(X_test)
    yt = np.array(y_test)

    model = KNN()
    model.fit(X, y)
    predictions = model.predict(Xt)
    return get_result(yt, predictions)


def knn_sklearn(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors=11,)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return get_result(y_test, predictions)
