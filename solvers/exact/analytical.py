from solvers import ISolver
from configuration import u
from diff import DiffSystem

import numpy as np


class AnalyticalSolution(ISolver):

    def __init__(self, original_function: callable = u):
        self.__func = original_function

    def solve(self, system: DiffSystem) -> np.ndarray:
        result = system.initial_state()

        for i in range(result.shape[1]):
            for j in range(result.shape[0]):
                result[j, i] = self.__func(x=system.space(j), t=system.time(i))

        return result
