from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix


def naive_bayes_sklearn(X, y, X_test, y_test):
    model = GaussianNB()
    model.fit(X, y)
    predictions = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions)
    result = {
        "f": f1_score(y_test, predictions, pos_label="h"),
        "recall": recall_score(y_test, predictions, pos_label="h"),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="h"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }
