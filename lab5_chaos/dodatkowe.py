import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

N=int(2e4)
c=0.05
f=0.3


def duff(y, t, a, b, c, f, omega):
    x, v = y
    dydt = [v, b*x-a*x**3-c*v+f*np.cos(omega*t)]
    return dydt

t = np.array(range(N))/0.2

sol = odeint(duff, [0, 0.11], t, args=(1, 1, c, f, 0.2*2*np.pi))

plt.scatter(sol[200:, 0], sol[200:, 1], marker='.',color='black',linewidth=0.1)
plt.xlabel("x")
plt.ylabel("v")
plt.title("Przecięcie Poincare równania Duffinga")
plt.savefig("poincare")