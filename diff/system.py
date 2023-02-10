from diff.mesh import Mesh

from scipy.integrate import quad
from typing import Dict
import numpy as np


class DiffSystem:
    def __init__(self, conditions: Dict[str, callable], space: dict, time: dict, approximation: callable = quad):

        if not self.check_system(conditions):
            raise ValueError("Not enough functions in system conditions!")

        self.__initial = conditions["initial"]
        self.__boundary = conditions["boundary"]
        self.__rhs = conditions["f"]

        self.__approximation = approximation

        self.__space = Mesh(space)
        self.__time = Mesh(time)

    @staticmethod
    def check_system(conditions: Dict[str, callable]) -> bool:
        if "initial" in conditions and "boundary" in conditions and "f" in conditions:
            return True

        return False

    def initial_state(self) -> np.ndarray:
        result = np.zeros((self.__space.steps, self.__time.steps))

        for i in range(result.shape[0]):
            result[i, 0] = self.__initial(i)

        return result

    def space(self, index: int) -> float:
        return self.__space[index]

    def get_space(self) -> np.ndarray:
        return self.__space.partition

    def time(self, index: int) -> float:
        return self.__time[index]

    def get_time(self) -> np.ndarray:
        return self.__time.partition

    def get_dt(self) -> float:
        return self.__time.step_size

    def enforce_boundary(self, C: np.ndarray, step: int) -> np.ndarray:
        C[0, step] = self.__boundary(0, step)
        C[-1, step] = self.__boundary(C.shape[0] - 1, step)

        return C

    def mass_matrix(self) -> np.ndarray:
        size = self.__space.steps
        result = np.zeros((size, size))

        for i in range(size):
            result[i][i] = 4
        for i in range(size - 1):
            result[i][i + 1] = 1
            result[i + 1][i] = 1

        return result * self.__space.step_size / 6

    def stiffness_matrix(self) -> np.ndarray:
        size = self.__space.steps
        result = np.zeros((size, size))

        for i in range(size):
            result[i][i] = 2
        for i in range(size - 1):
            result[i][i + 1] = -1
            result[i + 1][i] = -1

        return result * 1 / self.__space.step_size

    def load_vector(self, t: float) -> np.ndarray:
        # space_size = self.__space.steps
        #
        # result = np.zeros(space_size)
        #
        # for i in range(1, space_size):
        #     result[i] = self.__approximation(lambda x: self.__rhs(x, t), a=self.__space[i] - self.__space.step_size / 2,
        #                                      b=self.__space[i] + self.__space.step_size / 2)[0]
        #
        # return result[1:]
        return np.zeros(self.__space.steps)
