from abc import ABC, abstractmethod
import typing as t

import numpy as np


class AbstractDetector(ABC):

    @abstractmethod
    def process_batch(
            self,
            image: t.List[np.ndarray]
    ) -> t.List[t.List[t.Union[float, int, None]]]:
        ...
