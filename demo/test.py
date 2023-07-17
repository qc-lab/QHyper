from collections import namedtuple


mytup = namedtuple("mytup", [f"s{i}" for i in range(4)])

t = mytup(2,2, 4,44)

print(t)
