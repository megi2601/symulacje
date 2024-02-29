
import numpy as np
import matplotlib.pyplot as plt


N=10000

fractals = {"sierpinski": {"prob": np.array([1/3, 1/3, 1/3]),
                           "m": np.array([[0.5, 0, 0, 0.5, 0.25, np.sqrt(3.) / 4],
                                          [0.5, 0, 0, 0.5, 0, 0],
                                          [0.5, 0, 0, 0.5, 0.5, 0]])},
            "fern": {"prob": np.array([0.02, 0.09, 0.10, 0.79]),
                     "m": np.array([[0.001, 0.0, 0.0, 0.16, 0.0, 0.0],
                                    [-0.15, 0.28, 0.26, 0.24, 0.0, 0.44],
                                    [0.2, -0.26, 0.23, 0.22, 0.0, 1.6],
                                    [0.85, 0.04, -0.04, 0.85, 0.0, 1.6]])},
            "dragon": {"prob": np.array([0.787473, 0.212527]),
                       "m": np.array([[0.824074, 0.281482, -0.212346, 0.864198, -1.882290, -0.110607],
                                      [0.088272, 0.520988, -0.463889, -0.377778, 0.785360, 8.095795]])},
            "levy": {"prob": np.array([1/2, 1/2]),
                     "m": np.array([[0.5, -0.5, 0.5, 0.5, 0.0, 0.0],
                                    [0.5, 0.5, -0.5, 0.5, 0.5, 0.5]])
                     }}




def p(m, x, y):
    return m[0]*x+m[1]*y+m[4], m[2]*x+m[3]*y+m[5]

index_1 = np.random.randint(0, 3, N-1)
index_2 = np.random.choice([0, 1, 2, 3], N-1, p=fractals["fern"]["prob"])

def plot_fractal(x, y):
    fig, ax = plt.subplots()
    ax.set_facecolor('black')

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='both', length=0)
    plt.scatter(x, y, s=1, marker="o", lw=0, c="green")

    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig, ax


def fractal(name):
    f = fractals[name]
    x = np.zeros(N)
    y = np.zeros(N)
    index = np.random.choice(list(range(len(f["prob"]))), N-1, p=f["prob"])
    for n in range(1, N):
        xx, yy = p(f["m"][index[n-1]], x[n-1], y[n-1])
        x[n] = xx
        y[n] = yy
    plot_fractal(x, y)
    plt.savefig(name)
    return x, y


def plot_Nr(name):
    x, y = fractal(name)
    R = np.array(list(range(1, 15)))
    Nr = np.zeros(len(R))
    for r in R:
        H, xedges, yedges = np.histogram2d(x, y, np.power(2, r))
        Nr[r-1] = np.log(np.count_nonzero(H))
    plt.figure()
    plt.xlabel("r")
    plt.ylabel("log(Nr)")
    plt.title(name)
    plt.plot(R, Nr, 'or', label="calculated log(Nr)")
    [a, b], V = np.polyfit(R[:6], Nr[0:6], 1, cov=True)
    Df = a / np.log(2)
    delta_d = np.sqrt(V[0][0]) / np.log(2)
    plt.plot(R[:6], a*R[:6] + b*np.ones(6), label="fit")
    plt.text(8, 3, f'D_f: {Df:.4} \nniepewność: {delta_d:.2}')
    plt.legend()
    plt.savefig(name+"_Nr")


def full_plot():
    for name in fractals.keys():
        plot_Nr(name)


full_plot()
