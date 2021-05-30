import typing as t
import os

import numpy as np
import cv2
import torch

from object_detector.abstract_detector import AbstractDetector
from object_detector.yolov4.utils.darknet2pytorch import Darknet
from helpers import LoggerMixin


class YoloV4Detector(AbstractDetector, LoggerMixin):
    dependencies_loc = os.path.join(
        os.path.dirname(__file__), "dependencies"
    )

    def __init__(
            self,
            model_name: str,
            path_to_config: t.Optional[str] = None,
            path_to_weights: t.Optional[str] = None,
            path_to_txt: t.Optional[str] = None,
            device: str = "gpu"
    ) -> None:
        if path_to_config:
            self._path_to_config = path_to_config
        else:
            self._path_to_config = os.path.join(
                YoloV4Detector.dependencies_loc, "v4.cfg"
            )
        if path_to_weights:
            self._path_to_weights = path_to_weights
        else:
            self._path_to_weights = os.path.join(
                YoloV4Detector.dependencies_loc, "v4.weights"
            )
        if path_to_txt:
            self._path_to_txt = path_to_txt
        else:
            self._path_to_txt = os.path.join(
                YoloV4Detector.dependencies_loc, "classes.txt"
            )
        # Initialize and prepare the model
        try:
            self._model = Darknet(self._path_to_config)
        except Exception as e:
            raise Exception(f"Failed to init Darknet. Error: {e}")
        else:
            try:
                self._model.load_weights(self._path_to_weights)
            except Exception as e:
                raise Exception(f"Failed to load weights. Error: {e}")
        self._device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "cuda"
            else torch.device("cpu")
        )
        self._model.to(self._device).eval()

        self._classes = YoloV4Detector.load_classes(self._path_to_txt)
        self.logger.info(
            f"Detecting the following classes: {' '.join(self._classes)}"
        )
        self.logger.info(f"Model {model_name} initialized")

    def process_batch(
            self,
            image: t.List[np.ndarray]
    ) -> t.List[t.List[t.Union[float, int, None]]]:
        pass

    @staticmethod
    def load_classes(path_to_txt: str) -> t.List[str]:
        with open(path_to_txt, "r") as file:
            return [item.strip() for item in file.readlines()]
