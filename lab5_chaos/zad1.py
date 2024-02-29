import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt



def duff(y, t, a, b, c, f, omega):
    x, v = y
    dydt = [v, b*x-a*x**3-c*v+f*np.cos(omega*t)]
    return dydt

t = np.linspace(0, 100, 10000)

sol = odeint(duff, [0, 0.11], t, args=(1, 1, 0.2, 0.2, 0.4*np.pi))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].plot(t, sol[:, 0], 'b', label='x(t)')
axes[1].plot(t, sol[:, 1], 'g', label='v(t)')
axes[0].set_ylabel("x")
axes[1].set_ylabel("v")
axes[0].set_xlabel("t")
axes[1].set_xlabel("t")
fig.suptitle("Rozwiązanie równania Duffinga z f=0.2")
fig.savefig("Duffing_v_x")
fig, ax = plt.subplots()
ax.plot(sol[:, 0], sol[:, 1], label="(x(t), v(t))")
ax.set_xlabel('x')
ax.set_ylabel('v')
fig.suptitle("Rozwiązanie równania Duffinga z f=0.2")
fig.savefig("Duffing_v(x)")