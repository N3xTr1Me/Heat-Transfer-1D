from configuration import storage_path
import numpy as np

from matplotlib import pyplot
from seaborn import heatmap
from celluloid import Camera

import errno
import os


class Graphics:
    def __init__(self):
        self.__path = storage_path

        self.check_storage()

    def _storage_exists(self) -> bool:
        if os.path.exists(self.__path):
            return True

        return False

    def _create_storage(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.__path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    def check_storage(self) -> None:
        if not self._storage_exists():
            self._create_storage()

    def make_heatmap(self, solution: np.ndarray, name: str) -> None:
        heatmap(solution)

        self.check_storage()
        pyplot.savefig(self.__path + name + ".png")

        pyplot.plot()
        pyplot.clf()

    def make_gif(self, matrix: np.ndarray, space_steps: int, time_steps: int) -> None:
        self.check_storage()

        fig, ax = pyplot.subplots()
        camera = Camera(fig)
        test_E = matrix.T[::-1]
        x_vec = range(0, space_steps)

        for t in range(time_steps - 1, 0, -1):
            ax.set(xlabel="n", ylabel="temperature", title=f"Numerical Solution, dt=1/{time_steps}, dx=1/{space_steps}")
            pyplot.xlim([0, space_steps])
            pyplot.ylim([0, 1])
            pyplot.title = f"Numerical Solution, dt=1/{time_steps}, dx=1/{space_steps}"
            pyplot.plot(x_vec, test_E[t], 'black')
            camera.snap()

        animation = camera.animate()
        animation.save(self.__path + "heat_transfer.gif")
        pyplot.clf()

    def make_graphs(self, partition: np.ndarray, numeric: np.ndarray, exact: np.ndarray,
                    space_steps: int, time_steps: int) -> None:
        self.check_storage()

        fig, ax = pyplot.subplots()
        ax.plot(partition, numeric[:, -1], label='Numerical Solution', linewidth=6)
        ax.plot(partition, exact[:, -1], label='Analytical Solution', linewidth=3, linestyle='dashed')
        ax.set(xlabel="x", ylabel="temperature",
               title=f"Numeric and Analytical solutions. dt=1/{time_steps}, dx=1/{space_steps}")
        ax.legend()

        fig.savefig(self.__path + "numeric_and_exact.png")
        pyplot.show()
        pyplot.clf()

    def make_diff(self, partition: np.ndarray, numeric: np.ndarray, exact: np.ndarray,
                  space_steps: int, time_steps: int) -> None:

        self.check_storage()

        fig, ax = pyplot.subplots()
        diff = np.zeros(exact.shape)

        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                diff[i, j] = abs(numeric[i, j] - exact[i, j])

        ax.plot(partition, diff[:, -1])
        ax.set(xlabel="x", ylabel="temperature",
               title=f"Numeric/Analytical difference dt=1/{time_steps}, dx=1/{space_steps}")

        fig.savefig("./plots/numeric_exact_difference.png")
        pyplot.show()
        pyplot.clf()
