from abc import ABC
from abc import abstractmethod

import numpy as np


class AbstractAngleDetector(ABC):
    @abstractmethod
    def calculate_angle(self, img: np.array):
        pass
