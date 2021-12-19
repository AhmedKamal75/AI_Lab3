import random
import pandas as pd

dataset = pd.read_csv("data/magic04.data").to_numpy()

h = []
g = []
# others = 0


# for c in dataset[:, -1]:
#     if c == 'h':
#         h += 1
#     elif c == 'g':
#         g += 1
#     else:
#         others += 1
#
# print(f"# of h = {h}")
# print(f"# of g = {g}")
# print(f"# of others = {others}")


for c in range(len(dataset)):
    if dataset[c][-1] == 'h':
        h.append(dataset[c])
    elif dataset[c][-1] == 'g':
        g.append(dataset[c])


def data_balancing(a, b):
    if len(a) > len(b):
        return b, random.choices(a, k=len(b))
    else:
        return a, random.choices(b, k=len(a))


hard_on_list, gamma_list = data_balancing(h, g);

def data_splitting():