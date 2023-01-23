import numpy as np


class Mesh:
    def __init__(self, steps: int, start: float, end: float):
        self.steps = steps
        self.step_size = abs(end - start) / (steps - 1)

        self.partition = np.linspace(start=start, stop=end, num=steps)

    def __getitem__(self, item: int) -> float:
        return self.partition[item]
