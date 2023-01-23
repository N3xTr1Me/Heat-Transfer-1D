from abc import ABC, abstractmethod

from diff import Domain

import numpy as np


class ISolver(ABC):

    @abstractmethod
    def solve(self, domain: Domain) -> np.ndarray:
        pass
