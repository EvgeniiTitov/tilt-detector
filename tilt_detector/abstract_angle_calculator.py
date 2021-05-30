import typing as t
from abc import ABC
from abc import abstractmethod

import numpy as np


class AbstractAngleCalculator(ABC):
    @abstractmethod
    def calculate_inclination(
        self, image: np.array
    ) -> t.Tuple[t.Optional[float], t.Optional[list]]:
        pass
