import typing as t
from abc import ABC
from abc import abstractmethod

import numpy as np


class AbstractDetector(ABC):
    @abstractmethod
    def process_batch(
        self, batch: t.List[np.ndarray]
    ) -> t.List[t.List[t.Union[float, int, None]]]:
        ...
