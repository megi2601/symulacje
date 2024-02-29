import numpy as np
from matplotlib import pyplot as plt
import cmath

M = np.zeros((100, 100, 2))
N = np.zeros((100, 100))

c = np.array([-0.73, 0.27015])

def multiply(l):
    r = l[0]**2-l[1]**2
    i = 2*l[1]*l[0]
    if r>10000 or i>10000:
        return l
    return np.array([r, i])

for i in range(100):
    for j in range(100):
        M[i, j] = np.array([i, j])

for iter in range(1000):
    for i in range(100):
        for j in range(100):
            M[i, j] = multiply(M[i, j]) + c

for i in range(100):
    for j in range(100):
        N[i, j] = M[i, j][0]**2 - M[i, j][1]**2




plt.imshow(N)
plt.show()