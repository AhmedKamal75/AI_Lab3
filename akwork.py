import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from Naive_Bayes import naive_bayes_gaussian


def ready_data(dataset):
    inputs = np.array(dataset.values[:, 0:-1], dtype=np.float32)
    temp_target = dataset.values[:, -1]
    for i, j in enumerate(temp_target):
        if j == 'h':
            temp_target[i] = 0.0
        else:
            temp_target[i] = 1.0
    targets = np.array(temp_target, dtype=np.float16)
    targets = targets.reshape((targets.shape[0], 1))
    # under sampling
    # rus = RandomUnderSampler(random_state=0, replacement=True)
    # X_res, y_res = rus.fit_resample(inputs, targets)

    # splitting
    # random_state=0 means that every time we have the same set of selected
    # return train_test_split(X_res, y_res, test_size=0.3, random_state=0)
    return inputs, targets


def main(dataset, method="NB"):
    global result
    inputs, targets = ready_data(dataset)
    if method == "NB":
        # no need to resample any thing
        X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3, random_state=0)
        result = naive_bayes_gaussian(X_train, y_train, X_test, y_test)
    elif method == "DT":
        pass
    elif method == "RF":
        pass
    elif method == "AB":
        pass
    elif method == "KNN":
        pass

    for title in result:
        print(f"{title} --> {result.get(title)}")


if __name__ == "__main__":
    main(pd.read_csv("data/magic04.data"), "NB")
