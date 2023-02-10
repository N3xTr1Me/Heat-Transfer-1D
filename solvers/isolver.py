from abc import ABC, abstractmethod

from diff import DiffSystem

import numpy as np


class ISolver(ABC):

    @abstractmethod
    def solve(self, domain: DiffSystem) -> np.ndarray:
        pass
