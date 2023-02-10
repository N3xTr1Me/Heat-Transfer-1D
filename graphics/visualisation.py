from matplotlib import pyplot
from seaborn import heatmap
from celluloid import Camera

import numpy as np


def generate_gif(matrix: np.ndarray, n: int, h: int) -> None:
    fig, ax = pyplot.subplots()
    camera = Camera(fig)
    test_E = matrix.T[::-1]
    x_vec = range(0, n)
    for t in range(h - 1, 0, -1):
        ax.set(xlabel="n", ylabel="temperature", title=f"Numerical Solution, dt=1/{h - 1}, dx=1/{n - 1}")
        pyplot.xlim([0, n])
        pyplot.ylim([0, 1])
        pyplot.title = f"Numerical Solution, dt=1/{h - 1}, dx=1/{n - 1}"
        pyplot.plot(x_vec, test_E[t], 'black')
        camera.snap()

    animation = camera.animate()
    animation.save("./plots/heat_transfer.gif")


def generate_graphs(partition: np.ndarray, U: np.ndarray, E: np.ndarray, n: int, h: int):
    fig, ax = pyplot.subplots()
    ax.plot(partition, U[:, -1], label='Numerical Solution', linewidth=6)
    ax.plot(partition, E[:, -1], label='Analytical Solution', linewidth=3, linestyle='dashed')
    ax.set(xlabel="x", ylabel="Heat", title=f"Solution at t=1. dt=1/{h - 1}, dx=1/{n - 1}")
    # ax.legend()
    # ax.grid()
    fig.savefig("./plots/numeric-exact-difference.png")
    pyplot.show()


def generate_heatmap(solution: np.ndarray, name: str) -> None:
    heatmap(solution)
    pyplot.savefig("./plots/" + name + ".png")
    pyplot.plot()
    pyplot.clf()
