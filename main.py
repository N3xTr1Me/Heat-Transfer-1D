import matplotlib.pyplot as plt
import numpy as np
import seaborn

from diff import *
from graphics import generate_graphs, generate_gif
from solvers import EulerMethod, AnalyticalSolution
from configuration import space_conditions, time_conditions, system_conditions


def main():
    # # Creating the space and time partition of the given intervals
    # space = Mesh(space_steps, space_interval[0], space_interval[1])
    # time = Mesh(time_steps, time_interval[0], time_interval[1])
    #
    # # Constructing domain area
    # domain_area = Domain(space=space,
    #                      time=time)

    domain = DiffSystem(system_conditions, space_conditions, time_conditions)

    numeric = EulerMethod()
    exact = AnalyticalSolution()

    U = numeric.solve(domain)

    E = exact.solve(domain)

    generate_gif(E, space_conditions["steps"], time_conditions["steps"])
    generate_graphs(domain.get_space(), U, E, space_conditions["steps"], time_conditions["steps"])
    seaborn.heatmap(U[1:-1])
    plt.savefig("./plots/heatmap.png")
    plt.plot()

    # np.save("./plots/numeric_solution", U)

    # U1 = np.load("./plots/numeric_solution.npy")
    #
    # print(U - U1)


if __name__ == '__main__':
    main()
