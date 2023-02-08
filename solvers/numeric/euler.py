from configuration import is_explicit
from solvers import ISolver
from diff import Domain

import numpy as np


class EulerMethod(ISolver):
    def __init__(self, mode: bool = is_explicit):
        self.__mode = mode
        self._domain = None

    def _derivative(self, current_step: np.ndarray, t: float) -> np.ndarray:
        B = (self._domain.M + self._domain.dt() / 2 * self._domain.S)
        A = (self._domain.M - self._domain.dt() / 2 * self._domain.S)

        return np.linalg.inv(B) @ ((A - B) @ current_step + self._domain.get_load(t=t)) / self._domain.dt()

    def _forward(self, current_step: np.ndarray, t: float, dt: float) -> np.ndarray:
        return current_step + dt * self._derivative(current_step, t)

    def _backward(self, current_step: np.ndarray, t: float, dt: float) -> np.ndarray:
        dv = self._derivative(current_step, t)
        print(dv)
        return current_step + dt * dv

    def solve(self, domain: Domain) -> np.ndarray:
        self._domain = domain

        border = 5

        C = np.zeros((self._domain.space_steps(), self._domain.time_steps()))
        C[:, 0] = np.ones((self._domain.space_steps()))
        C[0, 0] = border
        C[-1, 0] = border

        step = 0

        time = self._domain.get_time()

        for t in time[:-1]:
            C[0, step] = border
            C[-1, step] = border

            if self.__mode:
                C[:, step + 1] = self._forward(current_step=C[:, step], t=t, dt=domain.dt())
            else:
                C[:, step + 1] = self._backward(current_step=C[:, step],
                                                t=self._domain.time(time.index(t) + 1), dt=domain.dt())

            step += 1

        C[0, -1] = border
        C[-1, -1] = border

        self._domain = None

        return C
