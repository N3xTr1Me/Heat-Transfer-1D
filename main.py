import numpy as np

from diff import *
from graphics import generate_graphs, generate_gif, generate_heatmap
from solvers import EulerMethod, AnalyticalSolution
from configuration import space_conditions, time_conditions, system_conditions


def main():

    domain = DiffSystem(system_conditions, space_conditions, time_conditions)

    numeric = EulerMethod()
    exact = AnalyticalSolution()

    U = numeric.solve(domain)

    E = exact.solve(domain)

    print("L2-norm difference between numeric and analytical solutions: ", np.linalg.norm(E - U) / E.size)

    generate_heatmap(U[1: -1], "numeric")
    generate_heatmap(E, "analytical")

    generate_graphs(domain.get_space(), U, E, space_conditions["steps"], time_conditions["steps"])
    generate_gif(U, space_conditions["steps"], time_conditions["steps"])


if __name__ == '__main__':
    main()
