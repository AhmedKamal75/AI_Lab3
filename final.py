import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dataset = pd.read_csv("data/magic04.data",
                      names=["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha",
                             "fDist", "Class"])

h = []
g = []

for r in range(len(dataset)):
    row = dataset.loc[r]
    if row.Class == "h":
        h.append(row)
    elif row.Class == "g":
        g.append(row)


def data_balancing(a, b):
    if len(a) > len(b):
        return b, random.choices(a, k=len(b))
    else:
        return a, random.choices(b, k=len(a))


hard_on_list, gamma_list = data_balancing(h, g);
all_data = hard_on_list + gamma_list;
balanced_dataset = pd.DataFrame(all_data)
X = balanced_dataset.drop('Class', axis=1)
y = balanced_dataset['Class']


def spliting_scaling():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# criterion :{“gini”, “entropy”},   default=”gini”
# splitter  :{“best”, “random”},    default=”best”
def decision_tree(criterion, splitter):
    X_train, X_test, y_train, y_test = spliting_scaling()
    classifier = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


# n_estimators  :int,   default=50
def ada_boost(n_estimators):
    X_train, X_test, y_train, y_test = spliting_scaling()
    classifier = AdaBoostClassifier(n_estimators=n_estimators)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


# n_neighbors :int,     default=5
def k_nn(n_neighbors):
    X_train, X_test, y_train, y_test = spliting_scaling()
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


# criterion :{“gini”, “entropy”},   default=”gini”
# n_estimators  :int,   default=100
def random_forests(criterion, n_estimators):
    X_train, X_test, y_train, y_test = spliting_scaling()
    classifier = RandomForestClassifier(criterion=criterion, n_estimators=n_estimators)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


def naive_bayes():
    X_train, X_test, y_train, y_test = spliting_scaling()
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


print("1)decision_tree()")
decision_tree("gini", "best")
print("------------------")
print("2)ada_boost()")
ada_boost(50)
print("------------------")
print("3)k_nn()")
k_nn(5)
print("------------------")
print("4)random_forests()")
random_forests("gini", 100)
print("------------------")
print("5)naive_bayes()")
naive_bayes()
print("------------------")
