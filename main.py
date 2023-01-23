from diff import *
from graphics import generate_graphs, generate_gif
from solvers import EulerMethod, AnalyticalSolution
from configuration import space_steps, time_steps, space_interval, time_interval


def main():
    # Creating the space and time partition of the given intervals
    space = Mesh(space_steps, space_interval[0], space_interval[1])
    time = Mesh(time_steps, time_interval[0], time_interval[1])

    # Constructing domain area
    domain_area = Domain(space=space,
                         time=time)

    numeric = EulerMethod()
    exact = AnalyticalSolution()

    U = numeric.solve(domain_area)
    E = exact.solve(domain_area)

    generate_gif(E, space_steps, time_steps)
    generate_graphs(space.partition, U, E, space_steps, time_steps)


if __name__ == '__main__':
    main()
