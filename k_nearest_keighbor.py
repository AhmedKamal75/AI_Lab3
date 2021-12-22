import numpy as np
import torch
from Naive_Bayes import get_result


def distance(point_1, point_2):
    return torch.sqrt(torch.sum((point_1 - point_2) ** 2)).item()


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.count = 0

    def fit(self, X, y):
        self.y = y
        self.X = X

    def predict(self, x):
        return np.array([self.predict_instance(instance_x) for instance_x in x])

    def predict_instance(self, x):
        self.count += 1
        distances = torch.tensor([distance(x, x_train) for x_train in self.X])
        indices_of_sorted_distances = torch.argsort(distances)[:self.k]
        nearest_labels = torch.tensor([self.y[i] for i in indices_of_sorted_distances])
        label = torch.mode(nearest_labels)[0].item()
        print(self.count / self.X.shape[0])
        return label


def knn(X_train, y_train, X_test, y_test):
    X = torch.tensor(X_train)
    y = torch.tensor(y_train)
    model = KNN()
    model.fit(X, y)
    predictions = model.predict(torch.tensor(X_test))
    return get_result(y_test, predictions)
