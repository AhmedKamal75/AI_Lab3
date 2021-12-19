import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from Naive_Bayes import naive_bayes_sklearn


def ready_data(X, y):
    # under sampling
    rus = RandomUnderSampler(random_state=0, replacement=True)
    X_res, y_res = rus.fit_resample(X, y)

    # splitting
    # random_state=0 means that every time we have the same set of selected
    return train_test_split(X_res, y_res, test_size=0.3, random_state=0)


def main(dataset, method="NB"):
    if method == "NB":
        # no need to resample any thing
        X_train, X_test, y_train, y_test = train_test_split(dataset.values[:, 0:-1], dataset.values[:, -1],
                                                            test_size=0.3,
                                                            random_state=0)
        result = naive_bayes_sklearn(X_train, y_train, X_test, y_test)
    else:
        X_train, X_test, y_train, y_test = ready_data(dataset.values[:, 0:-1], dataset.values[:, -1])


if __name__ == "__main__":
    main(pd.read_csv("data/magic04.data"), "NB")
