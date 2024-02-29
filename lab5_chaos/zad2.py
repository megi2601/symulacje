import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt



def duff(y, t, a, b, c, f, omega):
    x, v = y
    dydt = [v, b*x-a*x**3-c*v+f*np.cos(omega*t)]
    return dydt

t = np.linspace(0, 600, 60000)

l = {
    "podjedyczny okres, f= " : 0.2,
    "podwójny okres, f=" : 0.25,
    "poczwórny okres, f=" : 0.2675,
    "chaos, f=" : 0.28
}

for i in range(4):   
    sol = odeint(duff, [0, 0.11], t, args=(1, 1, 0.2, list(l.values())[i], 0.4*np.pi))
    plt.figure()
    plt.title("Rozwiązanie równania Duffinga - "+list(l.keys())[i]+str(list(l.values())[i]))
    plt.plot(sol[20000:, 0], sol[20000:, 1])
    plt.xlabel("x")
    plt.ylabel("v")
    plt.savefig(f"zad2_{i}")


