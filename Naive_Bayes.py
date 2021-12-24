import math
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
import torch


class NaiveBayesGaussian:
    def __init__(self):
        self.ones_num = 0
        self.zero_num = 0
        self.ones_mean = None
        self.zero_mean = None
        self.ones_std = None
        self.zero_std = None

    def fit(self, X, y):
        X = torch.tensor(X)
        y = torch.tensor(y)
        self.ones_num = torch.count_nonzero(y).item()
        self.zero_num = y.size()[0] - self.ones_num
        X_0 = torch.zeros((self.zero_num, X.shape[1]))
        X_1 = torch.zeros((self.ones_num, X.shape[1]))

        i = 0
        j = 0
        for col, cls in enumerate(y):
            if cls == 1.0:
                X_1[i, :] = X[col, :]
                i += 1
            else:
                X_0[j, :] = X[col, :]
                j += 1

        self.zero_mean = torch.mean(X_0, dim=0)
        self.ones_mean = torch.mean(X_1, dim=0)
        self.zero_std = torch.std(X_0, dim=0)
        self.ones_std = torch.std(X_1, dim=0)

    def predict(self, X):
        predictions = torch.ones((X.shape[0], 1))
        for i, instance in enumerate(X):
            if self.predict_instance(instance) == 0:
                predictions[i, 0] = 0
        return predictions.numpy()

    def predict_instance(self, x):
        score_0 = math.log(self.zero_num / (self.zero_num + self.ones_num))
        for (point, mean, std) in zip(x, self.zero_mean, self.zero_std):
            score_0 += norm.logpdf(point, loc=mean, scale=std)

        score_1 = math.log(self.ones_num / (self.zero_num + self.ones_num))
        for (point, mean, std) in zip(x, self.ones_mean, self.ones_std):
            score_1 += norm.logpdf(point, loc=mean, scale=std)

        if score_1 > score_0:
            return 1.0
        else:
            return 0.0


def naive_bayes_gaussian(X, y, X_test, y_test):
    model = NaiveBayesGaussian()
    model.fit(X, y)
    predictions = model.predict(X_test)

    return get_result(y_test, predictions)


def naive_bayes_sklearn(X, y, X_test, y_test):
    model = GaussianNB()
    model.fit(X, y)
    predictions = model.predict(X_test)

    return get_result(y_test, predictions)


def get_result(true, predictions):
    print(true)
    print(predictions)
    (tn, fp), (fn, tp) = confusion_matrix(true, predictions)
    result = {
        "f": f1_score(true, predictions, pos_label=1.0),
        "recall": recall_score(true, predictions, pos_label=1.0),
        "accuracy": accuracy_score(true, predictions),
        "precision": precision_score(true, predictions, pos_label=1.0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }
    return result
