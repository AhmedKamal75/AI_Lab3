import pandas as pd
import pandas as np

dataset = pd.read_csv("data/magic04.data").to_numpy()

h = 0
g = 0
others = 0

for c in dataset[:, -1]:
    if c == 'h':
        h += 1
    elif c == 'g':
        g += 1
    else:
        others += 1

print(f"# of h = {h}")
print(f"# of g = {g}")
print(f"# of others = {others}")
