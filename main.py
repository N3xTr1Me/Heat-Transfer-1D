import numpy as np

from diff import *
from graphics import Graphics
from solvers import EulerMethod, AnalyticalSolution
from configuration import space_conditions, time_conditions, system_conditions


def main():

    equation_system = DiffSystem(system_conditions, space_conditions, time_conditions)
    graphs = Graphics()

    numeric = EulerMethod()
    exact = AnalyticalSolution()

    U = numeric.solve(equation_system)

    E = exact.solve(equation_system)

    print("L2-norm mean difference between numeric and analytical solutions: ", np.linalg.norm(E - U) / E.size)

    graphs.make_heatmap(U[1: -1], "numeric")
    graphs.make_heatmap(E[1: -1], "analytical")

    graphs.make_graphs(equation_system.get_space(), U, E, space_conditions["steps"], time_conditions["steps"])
    graphs.make_diff(equation_system.get_space(), U, E, space_conditions["steps"], time_conditions["steps"])
    graphs.make_gif(U, space_conditions["steps"], time_conditions["steps"])


if __name__ == '__main__':
    main()
