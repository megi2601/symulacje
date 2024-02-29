import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def lorentz(Y, t, sigma, rho, beta):
    x, y, z = Y
    dYdt = [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]
    return dYdt

t = np.linspace(0, 50, 10000)

sol = odeint(lorentz, [1, 1, 1], t, args=(10, 28, 8/3))

fig = plt.figure(figsize=(12, 6))
grid = GridSpec(3, 2, width_ratios=[1, 1])
ax1 = plt.subplot(grid[0, 0])
ax2 = plt.subplot(grid[1, 0])
ax3 = plt.subplot(grid[2, 0])
ax1.plot(t, sol[:, 0])
ax2.plot(t, sol[:, 1])
ax3.plot(t, sol[:, 2])
ax1.set_xlabel("t")
ax1.set_ylabel("x")
ax2.set_xlabel("t")
ax2.set_ylabel("y")
ax3.set_xlabel("t")
ax3.set_ylabel("z")
ax4 = plt.subplot(grid[:, 1], projection='3d')
ax4.plot(sol[:, 0], sol[:,1], sol[:, 2])
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_zlabel("z")
fig.suptitle("System Lorentza")
plt.savefig("lorentz")