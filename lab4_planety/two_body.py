import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def F(r, m, G):
    indexes = range(int(len(r)/2))
    forces = np.zeros(len(r))
    for i in indexes:
        #  żeby wyszła elipsa
        if i==0:
            continue
        for j in indexes:
            if i!=j:
                l = np.sqrt((r[2*i]-r[2*j])**2+(r[2*i+1]-r[2*j+1])**2)
                forces[2*i] -= m[i]*m[j]*G*(r[2*i]-r[2*j]) /l/l/l
                forces[2*i+1] -= m[i]*m[j]*G*(r[2*i+1]-r[2*j+1]) /l/l/l
    return forces

def kinetic(p, m):
    m = np.repeat(m, 2)
    assert p.shape[1] == len(m)
    res = np.zeros(len(p))
    for i, pi in enumerate(p):
        res[i] = np.sum(pi*pi/2 /m)
    return res

def potential(r, m, G):
    indexes = range(int(r.shape[1]/2))
    pot = np.zeros(r.shape[0])
    for ind, ri in enumerate(r):
        for i in indexes:
            for j in indexes[i+1:]:
                l = np.sqrt((ri[2*i]-ri[2*j])**2+(ri[2*i+1]-ri[2*j+1])**2)
                pot[ind] -= m[i]*m[j]*G/l
    return pot

# symulacje przyjmują wektory r i p o wymiarze 2*(liczba ciał), zwracają liste takich wektorów 

def euler(r0, p0, dt, N, m, G):
    t = np.array([dt*n for n in range(N)])
    dim = len(r0)
    r = np.zeros(shape=(N, dim))
    p = np.zeros(shape=(N, dim))
    r[0] = r0
    p[0] = p0   
    for n in range(0, N-1):
        f = F(r[n],m, G)
        p[n+1] = p[n] + dt*f
        r[n+1] = r[n] + dt/np.repeat(m, 2)*p[n] + dt*dt/2/np.repeat(m, 2)*f
    return t, r, p

def verlet(r0, p0, dt, N, m, G):
    dim = len(r0)
    t = np.array([dt*n for n in range(N)])
    r = np.zeros(shape=(N, dim))
    p = np.zeros(shape=(N, dim))
    r_init = r0 - p0/np.repeat(m, 2)*dt
    r[0] = r0
    r[1] = 2*r0 - r_init + dt*dt/np.repeat(m, 2)*F(r0, m, G)
    p[0] = (r[1] -r_init) / 2 / dt *np.repeat(m, 2)
    for n in range(1, N-1):
        f = F(r[n], m, G)
        r[n+1] = 2*r[n] -r[n-1] + dt*dt/np.repeat(m, 2)*f
        p[n] = (r[n+1]-r[n-1]) / 2 / dt *np.repeat(m, 2)
    r_end = 2*r[N-1] - r[N-2] + dt*dt/np.repeat(m, 2)*F(r[N-1], m, G)
    p[N-1] = (r_end - r[N-2]) / 2 / dt *np.repeat(m, 2)
    return t, r, p

def frog(r0, p0, dt, N, m, G):
    dim = len(r0)
    t = np.array([dt*n for n in range(N)])
    r = np.zeros(shape=(N, dim))
    p = np.zeros(shape=(N, dim))
    r[0] = r0
    p[0] = p0 - F(r0, m, G) /2 *dt
    for n in range(0, N-1):
        f = F(r[n], m, G)   
        p[n+1] = p[n] + dt*f        # odpowiada t-dt/2
        r[n+1] = r[n] + p[n+1]/np.repeat(m, 2)*dt
    P = np.array([(p[a] + p[a+1]) / 2 for a in range(len(p)-1)])
    p_final = p[-1] + dt*F(r[-1], m, G)
    P = np.vstack((P, (p[-1]+p_final)/2))
    return t, r, P

def plot(t, r, p, m, G):
    fig, axes = plt.subplots(2, 2)
    for i in range(int(len(r.T)/2)):
        axes[0, 0].scatter(r.T[2*i], r.T[2*i+1], label=f"{i}")
    axes[0, 0].legend()
    axes[0, 1].plot(t, potential(r, m, G))
    axes[1, 0].plot(t, kinetic(p, m))
    axes[1, 1].plot(t, potential(r, m, G)+kinetic(p, m))
    axes[0, 0].set_title("trajektoria")
    axes[1, 1].set_title("Całkowita energia")
    axes[1, 0].set_title("Energia kinetyczna")
    axes[0, 1].set_title("Energia potencjalna")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 1].set_ylabel("Ek")
    axes[0, 1].set_xlabel("t")
    axes[1, 0].set_ylabel("Ep")
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_ylabel("E")
    axes[1, 1].set_xlabel("t")
    plt.show()

def zad1():
    m = [500, 0.1]
    dt = 0.001
    N=12000
    G=0.01
    r0=np.array([0, 0, 2, 0])
    p0=np.array([0, 0, 0, 0.1])
    t, r, p = euler(r0, p0, dt, N, m, G)
    plot(t, r, p, m, G)
    t, r, p = verlet(r0, p0, dt, N, m, G)
    plot(t, r, p, m, G)
    t, r, p = frog(r0, p0, dt, N, m, G)
    plot(t, r, p, m, G)

def zad2():
    m = [1, 1, 1]
    dt = 0.001
    N=2000
    G=1
    r0=np.array([0.97000436,-0.24308753, -0.97000436,0.24308753, 0, 0])
    p0=np.array([0.93240737/-2,0.86473146/-2,0.93240737/-2,0.86473146/-2,0.93240737,0.86473146])
    t, r, p = euler(r0, p0, dt, N, m, G)
    plot(t, r, p, m, G)
    t, r, p = verlet(r0, p0, dt, N, m, G)
    plot(t, r, p, m, G)
    t, r, p = frog(r0, p0, dt, N, m, G)
    plot(t, r, p, m, G)

zad1()