import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))