import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


G = 0.01
M = 500
m = 0.1

# r - wektor
def F(r):
    R = np.sqrt(r.dot(r))  #
    return -M*m*G * r / R/R/R


def kinetic(p):
    P = np.zeros(len(p))
    for i, el in enumerate(p):
        P[i] = el.dot(el)
    return P/m/2

def potential(r):
    R = np.zeros(len(r))
    for i, el in enumerate(r):
        R[i] = np.sqrt(el.dot(el))
    return -M*m*G / R

def euler(r0, p0, dt, N):
    t = np.array([dt*n for n in range(N)])
    r = np.zeros(shape=(N, 2))
    p = np.zeros(shape=(N, 2))
    r[0] = r0
    p[0] = p0   #global
    for n in range(0, N-1):
        f = F(r[n])
        p[n+1] = p[n] + dt*f
        r[n+1] = r[n] + dt/m*p[n] + dt*dt/2/m*f
    return t, r, p

def plot(t, r, p):
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].scatter(r.T[0], r.T[1])
    axes[0, 1].plot(t, potential(r))
    axes[1, 0].plot(t, kinetic(p))
    axes[1, 1].plot(t, potential(r)+kinetic(p))
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


def verlet(r0, p0, dt, N):
    t = np.array([dt*n for n in range(N)])
    r = np.zeros(shape=(N, 2))
    p = np.zeros(shape=(N, 2))
    r_init = r0 - p0/m*dt
    r[0] = r0
    r[1] = 2*r0 - r_init + dt*dt/m*F(r0)
    p[0] = (r[1] -r_init) / 2 / dt *m
    for n in range(1, N-1):
        f = F(r[n])
        r[n+1] = 2*r[n] -r[n-1] + dt*dt/m*f
        p[n] = (r[n+1]-r[n-1]) / 2 / dt *m
    r_end = 2*r[N-1] - r[N-2] + dt*dt/m*F(r[N-1])
    p[N-1] = (r_end - r[N-2]) / 2 / dt *m
    return t, r, p

def frog(r0, p0, dt, N):
    t = np.array([dt*n for n in range(N)])
    r = np.zeros(shape=(N, 2))
    p = np.zeros(shape=(N, 2))
    r[0] = r0
    p[0] = p0 - F(r0) /2 *dt
    for n in range(0, N-1):
        f = F(r[n])   
        p[n+1] = p[n] + dt*f        # odpowiada t-dt/2
        r[n+1] = r[n] + p[n+1]/m*dt
    P = np.array([(p[a] + p[a+1]) / 2 for a in range(len(p)-1)])
    print(len(P))
    p_final = p[-1] + dt*F(r[-1])
    P = np.vstack((P, (p[-1]+p_final)/2))
    return t, r, P

def zad1():
    M = 500
    m = 0.1
    dt = 0.001
    r0=np.array([2, 0])
    p0=np.array([0, 0.1])
    t, r, p = euler(r0, p0, dt, 1000)
    print(r)
    plot(t, r, p)

    t, r, p = verlet(r0, p0, dt, 12000)
    plot(t, r, p)

    t, r, p = frog(r0, p0, dt, 12000)
    plot(t, r, p)


zad1()