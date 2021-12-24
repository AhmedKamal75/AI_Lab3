import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from Naive_Bayes import naive_bayes_gaussian
from adaboost import adaboost_classifier
from k_nearest_keighbor import knn_


def ready_data(dataset, label_encode=True, resample=True, split=True):
    inputs = np.array(dataset.values[:, 0:-1], dtype=np.float32)
    temp_targets = dataset.values[:, -1]
    # label encode the target
    if label_encode:
        temp_targets = LabelEncoder().fit_transform(temp_targets)

    targets = np.array(temp_targets, dtype=np.float16)
    targets = targets.reshape((targets.shape[0], 1))

    # under sampling
    if resample:
        rus = RandomUnderSampler(random_state=0, replacement=True)
        inputs, targets = rus.fit_resample(inputs, targets)
    # splitting
    # random_state=0 means that every time we have the same set of selected
    return train_test_split(inputs, targets, test_size=0.3, random_state=0) if split else (inputs, targets)


def main(dataset, method="NB"):
    global result
    if method == "NB":
        X_train, X_test, y_train, y_test = ready_data(dataset, resample=False)
        result = naive_bayes_gaussian(X_train, y_train, X_test, y_test)
    elif method == "DT":
        pass
    elif method == "RF":
        pass
    elif method == "AB":
        X_train, X_test, y_train, y_test = ready_data(dataset)
        result = adaboost_classifier(X_train, y_train, X_test, y_test)
    elif method == "KNN":  # very slow O(n*m) per sample, and we have 4000 sample. n = 9000, m = 10
        X_train, X_test, y_train, y_test = ready_data(dataset)
        result = knn_(X_train, y_train, X_test, y_test)
    else:
        return False

    for title in result:
        print(f"{title} --> {result.get(title)}")


if __name__ == "__main__":
    start = time.time()
    main(pd.read_csv("data/magic04.data"), "KNN")
    end = time.time()
    print(f"time taken: {end - start}")
