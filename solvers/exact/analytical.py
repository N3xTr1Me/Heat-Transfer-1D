from solvers import ISolver
from configuration import u
from diff import Domain

import numpy as np


class AnalyticalSolution(ISolver):

    def __init__(self, original_function: callable = u):
        self.__func = original_function

    def solve(self, domain: Domain) -> np.ndarray:
        space_steps = domain.space_steps()
        time_steps = domain.time_steps()
        result = np.zeros((space_steps, time_steps))

        for i in range(time_steps):
            for j in range(space_steps):
                result[j, i] = self.__func(x=domain.space(j), t=domain.time(i))

        return result
