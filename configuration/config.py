from math import pi, exp, sin
import numpy as np


# path to the directory, where the graphic plots will be saved
storage_path = "./plots/"

# space mesh conditions
space_conditions = {
    "interval": (0, 1),    # space partition interval borders
    "steps":50,          # number of elements in partition
    "step-size": None       # size of element on the mesh
}

# time mesh conditions
time_conditions = {
    "interval": (0, 1),    # time partition interval borders
    "steps": 1000,          # number of elements in partition
    "step-size": None       # size of element on the mesh
}

# ----------------------------------------------------------------------------------------------------------------------

# explicit / implicit method switch
is_explicit = False

# temperature on the borders of the rod
t_border = 0

# temperature inside the borders of the rod
t_inside = 1

# temperatures in each element of the rod partition at t = 0
initial_conditions = np.array([t_inside if x not in [0, space_conditions["steps"] - 1] else
                               t_border for x in range(space_conditions["steps"])])


system_conditions = {
    # Initial conditions function
    "initial": lambda x: initial_conditions[x],
    # Boundary conditions function
    "boundary": lambda x, t: t_border if x in [0, space_conditions["steps"] - 1] else t_inside,
    # Right hand side function
    "f": lambda x, t: 0
}


# original function for analytical solution
def u(x: float, t: float, count = 1000) -> float:
    result = 0
    right_border = space_conditions["interval"][1]

    for i in range(1, count + 1):
        result += (t_inside * 2 * (1 - pow(-1, i)) / (i * pi)) \
                  * sin(i * pi * x / right_border) * exp(-t * (i * pi / right_border) ** 2)

    return result
