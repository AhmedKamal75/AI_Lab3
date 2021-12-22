import math
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
import numpy as np
import torch

import Naive_Bayes


# def weighted_impurity(true_correct, true_wrong, false_correct, false_wrong):
#     gini_true_leaf = 1 - (true_correct / (true_wrong + true_correct)) ** 2 - (
#             true_wrong / (true_wrong + true_correct)) ** 2
#     gini_false_leaf = 1 - (false_correct / (false_wrong + false_correct)) ** 2 - (
#             false_wrong / (false_wrong + false_correct)) ** 2
#
#     true_weight = (true_correct + true_wrong) / (true_correct + true_wrong + false_correct + false_wrong)
#     false_weight = (false_correct + false_wrong) / (true_correct + true_wrong + false_correct + false_wrong)
#
#     return gini_true_leaf * true_weight + gini_false_leaf * false_weight

def weighted_impurity(counts):
    total_weighted_gini = 0
    sums = np.zeros(len(counts))
    for i, (leaf, _) in enumerate(counts):
        sums[i] = np.sum(leaf)
    for i, (leaf, _) in enumerate(counts):
        gini = 1
        for j in leaf:
            gini -= (j / sums[i]) ** 2
        weight = sums[i] / np.sum(sums)
        total_weighted_gini += gini * weight
    return total_weighted_gini


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


def count(x, y):
    if is_numeric(x):
        merged_data = np.transpose(np.vstack((x.numpy(), y.numpy())))
        sorted_data = torch.tensor(merged_data[merged_data[:, 0].argsort()])

        classes = np.array(list(set(y.numpy())))
        leaf = np.zeros(len(set(y.numpy())))

        possible_gini_counts = []

        for i in range(len(sorted_data) - 1):
            avg = (sorted_data[i, 0] + sorted_data[i + 1, 0]) / 2
            count_stump = [(leaf.copy(), u) for u in [f"x {_} {avg}" for _ in ('<', '>=')]]

            for instance in sorted_data:
                cls_index = np.where(classes == instance[1].item())[0][0]
                if instance[0].item() < avg:
                    count_stump[0][0][cls_index] += 1
                else:
                    count_stump[1][0][cls_index] += 1
            possible_gini_counts.append((weighted_impurity(count_stump), count_stump))
        avg_cutoff, counts = sorted(possible_gini_counts, key=lambda record: record[0])[0]
        return avg_cutoff, counts

    else:
        uniques = np.array(list(set(x.numpy())))
        classes = np.array(list(set(y.numpy())))

        leaf = np.zeros(classes.shape)
        s = [(leaf.copy(), u) for u in uniques]

        for i, cls in zip(x, y):
            cls_index = np.where(classes == cls.item())[0][0]
            unq_index = np.where(uniques == i.item())[0][0]
            s[unq_index][0][cls_index] += 1  # [the class position][the leaf/cls position][the unique position]

        return weighted_impurity(s), s


class Stump:
    def __init__(self, X, y, weights, root_feature=0):
        # self._true_correct = 0
        # self._true_wrong = 0
        # self._false_correct = 0
        # self._false_wrong = 0
        # self.x =
        # self.y =
        self.weight = weights
        self._total_gini, self._counts = count(X[:, root_feature], y)
        # self.total_gini = weighted_impurity(self.counts)

    @property
    def total_gini(self):
        return self._total_gini

    @property
    def counts(self):
        return self._counts

    # @property
    # def true_correct(self):
    #     return self._true_correct
    #
    # @true_correct.setter
    # def true_correct(self, true_correct):
    #     self._true_correct = true_correct
    #
    # @property
    # def true_wrong(self):
    #     return self._true_wrong
    #
    # @true_wrong.setter
    # def true_wrong(self, true_wrong):
    #     self._true_wrong = true_wrong
    #
    # @property
    # def false_correct(self):
    #     return self._false_correct
    #
    # @false_correct.setter
    # def false_correct(self, false_correct):
    #     self._false_correct = false_correct
    #
    # @property
    # def false_wrong(self):
    #     return self._false_wrong
    #
    # @false_wrong.setter
    # def false_wrong(self, false_wrong):
    #     self._false_wrong = false_wrong


class adaboost:
    def __init__(self):
        pass

    def fit(self, X, y):
        X = torch.tensor(X)
        y = torch.tensor(y)
        data_weights = torch.empty((X.shape[0], 1)).fill_(1 / X.shape[0])
        stump = self.get_best_stump(X, y, data_weights)
        for i in range(X.shape[1]):
            stump = Stump(X, y, data_weights, root_feature=i)

    def get_best_stump(self, X, y, data_weights):
        stumps = []

    def predict(self, X_test):
        pass


def adaboost_classifier(X, y, X_test, y_test):
    module = adaboost()
    module.fit(X, y)
    # predictions = module.predict(X_test)
    # return Naive_Bayes.get_result(y_test, predictions)
