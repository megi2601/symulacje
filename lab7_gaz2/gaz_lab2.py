import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import imageio
from numba import jit


N = 16

sigma = 1.0
epsilon = 1.0
rc = 2.5*sigma
dt = 0.001

temp = 2.5
kb=1

per_row = int(np.sqrt(N))
box_size = 2*per_row
R = sigma/2
steps = 10000
pic_step = 100


class czastka:
    """Klasa opisujaca pojedyncza czastke gazu
    """
    def __init__(self, radius, pos, vel):
        self.radius = radius
        self.r = pos 
        self.v = vel 
        self.v_minus = 0


#siła z jaką cząstka 2 działa na cząstkę 1
#@jit(nopython=True)
def F(r1, r2):
    images = [r2]
    x=0
    y=0
    if r1[0] < r2[0]:
        x=-1
    elif r1[0] > r2[0]: 
        x=1
    if r1[1] < r2[1]:
        y=-1
    elif r1[1] > r2[1]: 
        y=1
    images.append(np.array([r2[0]+x*box_size, r2[1]]))
    images.append(np.array([r2[0], r2[1]+y*box_size]))
    images.append(np.array([r2[0]+x*box_size, r2[1]+y*box_size]))
    rs = [np.sqrt(np.dot(r1-ri, r1-ri)) for ri in images]
    r = min(rs)
    r2 = images[rs.index(r)]
    if r > rc:
        return np.zeros(len(r1)), 0, 0
    f = 48 * epsilon / sigma / sigma * ((sigma / r)**14 -0.5 * (sigma / r)**8)*(r1-r2)
    pressure_comp = np.dot(-f, r1-r2)
    pot_comp = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
    return f, pressure_comp, pot_comp


def calculate_forces(molecules):
    forces = np.array([np.zeros(2) for _ in range(N)])
    pot = 0
    pressure_comps = 0
    for i in range(N):
        for j in range(N):
            if i==j:
                continue
            f, pressure_comp, pot_comp = F(molecules[i].r, molecules[j].r)
            forces[i] += f
            if i<j:
                pot += pot_comp
                pressure_comps+=pressure_comp
    return forces, pot, pressure_comps
    
def v2(molecules):
    res = 0
    for c in molecules:
        res+= np.dot(c.v, c.v)
    return res

def v2_minus(molecules):
    res = 0
    for c in molecules:
        res+= np.dot(c.v_minus, c.v_minus)
    return res       

def rysuj(molecules, i):
    plt.clf() # wyczyść obrazek
    fig = plt.gcf() # zdefiniuj nowy
    for p in molecules: # pętla po cząstkach
        a = plt.gca()
        cir = Circle((p.r[0], p.r[1]), radius = p.radius) # kółko tam gdzie jest cząstka
        a.add_patch(cir) # dodaj to kółko do rysunku
        a.plot() # narysuj
    plt.xlim((0, box_size)) # obszar do narysowania
    plt.ylim((0, box_size))
    fig.set_size_inches((6, 6)) # rozmiar rysunku
    plt.title(f'Symulacja gazu Lennarda-Jonesa, krok {i:06d}')
    plt.savefig(f'img{i:06d}.png')

def plot_param(steps, params, name):
    # plt.clf()
    # fig = plt.gcf() # zdefiniuj nowy
    plt.plot(steps ,params, label=name)
    # plt.title(name)
    # plt.savefig(name)

def evolve(temp):
    time = np.array(range(0, steps))[::pic_step]
    temperature = np.array([temp])
    r0 = np.array([np.array([1+i*box_size/per_row, 1+j*box_size/per_row]) for i in range(per_row) for j in range(per_row)])
    v0 = np.array([np.random.rand(2) for i in range(N)])
    Vcm = np.array(sum(v0)) / N
    v0 = np.array([v-Vcm for v in v0])
    a = np.sqrt(2*N*temp*kb / np.sum(v0*v0))
    v0 *= a
    molecules = []

    #tworzenie cząstek
    for i in range(N):
        c = czastka(R, r0[i], v0[i])
        molecules.append(c)   

    #parametry ułożenia początkowego
    f0, pot0, pc0 = calculate_forces(molecules)
    pressure = np.array([N*kb*temp/box_size/box_size + 1/2/box_size/box_size*pc0]) #ciśnienie
    energy = np.array([v2(molecules) / 2 +pot0])

    v0_minus = v0 - f0/2*dt
    for i in range(N):               #init v od -1/2
        molecules[i].v_minus = v0_minus[i]
    
    rysuj(molecules, 0)
    

    #ewolucja czasowa
    for step in range(1, steps):
        forces, pot, pc = calculate_forces(molecules)
        tau = 0
        for i, f in enumerate(forces):
            c = molecules[i]
            v_mi = c.v_minus + dt*f/2
            tau += np.dot(v_mi, v_mi) / 2/ N/ kb 
        eta = np.sqrt(temp/tau)
        for i, f in enumerate(forces):
            c = molecules[i]
            v_nowa = (2*eta-1)*c.v_minus + eta*dt*f
            c.v = 0.5*(c.v_minus+v_nowa)
            c.v_minus = v_nowa
            c.r += c.v_minus*dt
            c.r %= box_size
        #tworzenie wykresów
        if step % pic_step ==0:
            rysuj(molecules, step)
            kinetic = v2(molecules)
            temperature = np.append(temperature, kinetic/ N / kb /2)
            energy = np.append(energy, kinetic/2+pot)
            pressure = np.append(pressure, N*kb*temp/box_size/box_size + 1/2/box_size/box_size*pc)
    plt.clf()
    fig = plt.gcf() # zdefiniuj nowy
    print(temperature)
    plot_param(time, temperature, "Temperatura")
    plot_param(time, energy, "Energia")
    plot_param(time, pressure, "Ciśnienie")
    plt.legend()
    plt.title("Parametry")
    plt.savefig("Parametry")




evolve(temp)

filenames = [f'img{pic_step*i:06d}.png' for i in range(1, int(steps/pic_step))]

with imageio.get_writer('movie3.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)