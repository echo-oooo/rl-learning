import numpy as np


p = np.zeros((4, ))
p[2] = 1.0

v = [np.random.choice(4, p=p) for _ in range(100)]
print(v)