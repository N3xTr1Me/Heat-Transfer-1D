import numpy as np


# space mesh conditions
space_conditions = {
    "interval": (0, 10),    # space partition interval borders
    "steps": 100,          # number of elements in partition
    "step-size": None       # size of element on the mesh
}

# time mesh conditions
time_conditions = {
    "interval": (0, 5),    # time partition interval borders
    "steps": 500,          # number of elements in partition
    "step-size": None       # size of element on the mesh
}

# ----------------------------------------------------------------------------------------------------------------------

# explicit / implicit method switch
is_explicit = True

# temperature on the borders of the rod
t_border = 10

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
def u(x: float, t: float) -> float:
    return t_border if x in [0, space_conditions["steps"] - 1] else t_inside

