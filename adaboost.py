import math
import numpy as np
import dill
import Naive_Bayes

FILE_NAME = "data/adaboost_stumps.pickle"


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


def count(x, y):
    # if is_numeric(x):
    merged_data = (np.vstack((x, y))).transpose()
    sorted_data = merged_data[merged_data[:, 0].argsort()]

    classes = np.array(sorted(set(y)))
    leaf = np.zeros(classes.shape)
    possible_gini_counts = []

    partition = len(sorted_data) / 100
    for part in np.arange(partition, 99 * partition, partition):
        part = int(part)
        avg = (sorted_data[part, 0] + sorted_data[part + int(partition), 0]) / 2
        count_stump = [(leaf.copy(), answer) for answer in [True, False]]

        for instance in sorted_data:
            cls_index = np.where(classes == instance[1])[0][0]
            if instance[0] < avg:
                # true:: left branch
                count_stump[0][0][cls_index] += 1
            else:
                # false:: right branch
                count_stump[1][0][cls_index] += 1

        possible_gini_counts.append([weighted_impurity(count_stump), avg, count_stump])
        # print(f"\t\t stump building in progress: {(part / (100 * partition)):.2%}")

    gini_avg_cutoff, separation_point, counts = sorted(possible_gini_counts, key=lambda record: record[0])[0]
    return (gini_avg_cutoff, separation_point), (counts, classes)


class Stump:
    def __init__(self, X, y, weights, root_feature=0):
        # self._true_correct = 0
        # self._true_wrong = 0
        # self._false_correct = 0
        # self._false_wrong = 0
        # self.x =
        # self.y =
        # self.total_gini = weighted_impurity(self.counts)
        # build the stump
        print(f"\tstart creating trial stump {root_feature}#")
        self._root_feature = root_feature
        (self._total_gini, self._separation_point), (self._counts, self._classes) = count(X[:, root_feature], y)
        self._stump_weight = self.calc_stump_weight(X[:, root_feature], y, weights)
        self._new_weights = self.get_new_weights(X[:, root_feature], y, weights)
        print(f"\tfinish creating trial stump {root_feature}#")

    @property
    def new_weights(self):
        return self._new_weights

    @property
    def stump_weight(self):
        return self._stump_weight

    @property
    def root_feature(self):
        return self._root_feature

    @property
    def total_gini(self):
        return self._total_gini

    @property
    def counts(self):
        return self._counts

    @property
    def classes(self):
        return self._classes

    @property
    def separation_point(self):
        return self._separation_point

    def __str__(self) -> str:
        string = f"root feature: {self.root_feature}\n" \
                 f"total gini  : {self.total_gini}\n" \
                 f"weight      : {self.stump_weight}\n" \
                 f"separation  : {self.separation_point}\n" \
                 f"classes     : {self.classes}\n" \
                 f"{self.counts}\n"
        return string

    def __repr__(self) -> str:
        return self.__str__()

    def _predict(self, x):
        index = 1
        if x < self.separation_point:
            index = 0
        side = self.classes[int(self.counts[index][0].argsort()[-1])]
        return side

    # predict an instance using the stump
    def predict(self, x):
        x_instance = x[self.root_feature]
        return self._predict(x_instance)

    def calc_stump_weight(self, x, y, weights, error_term=1e-15):
        total_error = sum([((self._predict(x[i]) != y[i]) and weights[i]) for i in range(len(y))])
        new_weights = 0.5 * math.log((1 - total_error + error_term) / (total_error + error_term))
        return new_weights

    def get_new_weights(self, x, y, weights):
        new_weights = np.copy(weights)
        exponents = np.array(
            [(-1. if (self._predict(x[i]) == y[i]) else 1.) for i in range(len(weights))]).reshape(
            (len(weights), 1)) * self.stump_weight
        new_weights = new_weights * (math.e ** exponents)
        return new_weights / new_weights.sum()


def get_best_stump(X, y, data_weights, features_not_included):
    print(f"create the {len(features_not_included)}# stump")
    stumps = {}
    for root_feature in (set(range(X.shape[1])) - set(features_not_included)):
        stump = Stump(X, y, data_weights, root_feature=root_feature)
        stumps[stump.total_gini] = stump
    chosen = stumps.get(sorted(stumps)[0])
    return chosen


class Adaboost:
    def __init__(self):
        self.stumps = []
        self.obtained_features = []

    def fit(self, X, y, write_to_file=False, read_of_file=False):
        if read_of_file:
            with open(FILE_NAME, "rb") as file:
                i = 0
                running = True
                try:
                    while running:
                        (root_feature, best_stump) = dill.load(file)
                        self.stumps.append(best_stump)
                        self.obtained_features.append(root_feature)
                        i += 1
                except EOFError:
                    running = False
                    print(f"number of stumps loaded: {i}")
            for stump in self.stumps:
                print(stump)
        else:
            if write_to_file:
                with open(FILE_NAME, "wb"):
                    pass
            X = np.array(X)
            y = np.array(y)
            data_weights = np.ones((X.shape[0], 1)) * (1 / X.shape[0])
            for i in range(X.shape[1]):
                print(f"module is {len(self.obtained_features) / X.shape[1]:.2%} finished")
                best_stump = get_best_stump(X, y, data_weights, self.obtained_features)
                self.stumps.append(best_stump)
                self.obtained_features.append(best_stump.root_feature)
                data_weights = np.copy(best_stump.new_weights)
                print(best_stump.__str__())
                if write_to_file:
                    with open(FILE_NAME, "ab") as file:
                        dill.dump((best_stump.root_feature, best_stump), file=file)

    def predict(self, X_test):
        return np.array([(self._predict(instance)) for instance in X_test])

    def _predict(self, instance):
        weight_sum_ones = 0
        weight_sum_zero = 0
        for stump in self.stumps:
            if stump.predict(instance) == 1.0:
                weight_sum_ones += stump.stump_weight
            else:
                weight_sum_zero += stump.stump_weight
        if weight_sum_ones > weight_sum_zero:
            return 1.0
        else:
            return 0.0
        # return stats.mode(np.array([stump.predict(instance) for stump in self.stumps]))[0][0]


def adaboost_classifier(X, y, X_test, y_test, read=True):
    module = Adaboost()
    module.fit(X, y, read_of_file=read)
    predictions = module.predict(X_test)
    # right = 0
    # wrong = 0
    # for true, prediction in zip(y_test, predictions):
    #     if true == prediction:
    #         right += 1
    #     else:
    #         wrong += 1
    # print(f"right: {right}, wrong: {wrong}")
    return Naive_Bayes.get_result(y_test, predictions)
