from diff import DiffSystem
from solvers import ISolver

import numpy as np


class RungeKutta(ISolver):
    def __init__(self):
        self._domain = None

        self.__M = None
        self.__S = None

    def _derivative(self, current_step: np.ndarray, t: float) -> np.ndarray:
        B = (self._domain.mass_matrix() + self._domain.get_dt() / 2 * self._domain.stiffness_matrix())
        A = (self._domain.mass_matrix() - self._domain.get_dt() / 2 * self._domain.stiffness_matrix())

        return np.linalg.inv(B) @ ((A - B) @ current_step + self._domain.load_vector(t=t)) / self._domain.get_dt()

    def solve(self, domain: DiffSystem) -> np.ndarray:
        self._domain = domain

        C = domain.initial_state()

        step = 0

        time = self._domain.get_time()

        for t in time[:-1]:

            h = domain.get_dt() / 2

            domain.enforce_boundary(C, step)

            k1 = self._derivative(current_step=C[:, step], t=t)
            k2 = self._derivative(current_step=C[:, step] + h * k1, t=t + h)
            k3 = self._derivative(current_step=C[:, step] + h * k2, t=t + h)
            k4 = self._derivative(current_step=C[:, step] + domain.get_dt()  * k3, t=t + domain.get_dt())

            C[:, step + 1] = h / 3 * (k1 + 2 * k2 + 2 * k3 + k4)

        domain.enforce_boundary(C, len(time) - 1)

        self._domain = None

        return C
