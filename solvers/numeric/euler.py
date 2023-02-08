from configuration import is_explicit
from solvers import ISolver
from diff import DiffSystem

import numpy as np


class EulerMethod(ISolver):
    def __init__(self, mode: bool = is_explicit):
        self.__mode = mode

        self.__M = None
        self.__S = None

    def _derivative(self, current_step: np.ndarray, load: np.ndarray, dt: float) -> np.ndarray:
        B = (self.__M + dt / 2 * self.__S)
        A = (self.__M - dt / 2 * self.__S)

        return np.linalg.inv(B) @ ((A - B) @ current_step + load) / dt

    def _forward(self, current_step: np.ndarray, load_vector: np.ndarray, dt: float) -> np.ndarray:
        return current_step + dt * self._derivative(current_step, load_vector, dt)

    def _backward(self, current_step: np.ndarray, load_vector: np.ndarray, dt: float) -> np.ndarray:
        return current_step + dt * self._derivative(current_step, load_vector, dt)

    def _reset(self) -> None:
        self.__M = None
        self.__S = None

    def solve(self, system: DiffSystem) -> np.ndarray:

        self.__M = system.mass_matrix()
        self.__S = system.stiffness_matrix()

        C = system.initial_state()
        time = system.get_time().tolist()
        dt = system.get_dt()

        step = 0

        for t in time[:-1]:
            system.enforce_boundary(C, step)

            if self.__mode:
                C[:, step + 1] = self._forward(current_step=C[:, step], load_vector=system.load_vector(t=t), dt=dt)
            else:
                C[:, step + 1] = self._backward(current_step=C[:, step],
                                                load_vector=system.load_vector(system.time(time.index(t) + 1)), dt=dt)

            step += 1

        system.enforce_boundary(C, len(time) - 1)

        self._reset()

        return C
