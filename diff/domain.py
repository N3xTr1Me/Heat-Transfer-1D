from diff import Mesh


from scipy.integrate import quad
import numpy as np


class Domain:
    def __init__(self, space: Mesh, time: Mesh):

        self.__space = space
        self.__time = time

        self.__approximation = quad

        self.M = self._build_mass()
        self.S = self._build_stiffness()

    # time interval step size
    def dt(self) -> float:
        return self.__time.step_size

    # space interval step size
    def ds(self) -> float:
        return self.__space.step_size

    # number of steps on the time interval
    def time_steps(self) -> int:
        return self.__time.steps

    # number of steps on the space interval
    def space_steps(self) -> int:
        return self.__space.steps

    def time(self, index: int) -> float:
        return self.__time[index]

    def get_time(self) -> list:
        return self.__time.partition.tolist()

    def space(self, index: int) -> float:
        return self.__space[index]

    # Basic functionality
    # ------------------------------------------------------------------------------------------------------------------
    # Linear system supplement functions

    def _build_mass(self) -> np.ndarray:
        size = self.__space.steps
        result = np.zeros((size, size))

        for i in range(size):
            result[i][i] = 4
        for i in range(size - 1):
            result[i][i + 1] = 1
            result[i + 1][i] = 1

        return result * self.ds() / 6

    def _build_stiffness(self) -> np.ndarray:
        size = self.__space.steps
        result = np.zeros((size, size))

        for i in range(size):
            result[i][i] = 2
        for i in range(size - 1):
            result[i][i + 1] = -1
            result[i + 1][i] = -1

        return result * 1 / self.ds()

    def get_load(self, t: float) -> np.ndarray:

        space_size = self.__space.steps

        # result = np.zeros(space_size)

        # for i in range(1, space_size):
        #     # result[i] = self.__approximation(x=self.__space[i], t=t, h=self.ds())
        #     result[i] = self.__approximation(lambda x: f(x, t), a=self.__space[i] - self.ds() / 2,
        #                                       b=self.__space[i] + self.ds() / 2)[0]

        # return result[1:]
        return np.zeros(space_size)
