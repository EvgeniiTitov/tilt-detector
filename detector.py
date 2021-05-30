import os
import typing as t

import cv2
import numpy as np

from config import Config
from helpers import LoggerMixin
from helpers import ResultProcessor
from object_detector import YoloV4Detector
from tilt_detector import PoleAngleCalculator


class Detector(LoggerMixin):
    def __init__(
        self, source_folder: str, dest_folder: str, to_json: bool
    ) -> None:
        self._source = source_folder
        self._destination = dest_folder
        self._to_json = to_json
        self._detector_model = YoloV4Detector("obj_detector")
        self._tilt_detector = PoleAngleCalculator(True)
        self._result_processor = ResultProcessor()
        self.logger.info("Detector initialized")

    def process_images_in_folder(self):
        results = dict()
        for item in os.listdir(self._source):
            item_path = os.path.join(self._source, item)
            if os.path.isdir(item_path) or not any(
                item.lower().endswith(ext) for ext in Config.ALLOWED_EXTS
            ):
                self.logger.info(
                    f"Cannot process {item}. Unsupported extension"
                )
                continue

            image = cv2.imread(item_path)
            if image is None:
                self.logger.error(f"Failed to open image: {item}")
                continue
            # Run the model and keep only detected poles (insulators and
            # dumpers are discarded)
            predictions = [
                pred
                for pred in self._detector_model.process_batch([image])[0]
                if pred[-1] == "pole"
            ]
            if not len(predictions):
                self._result_processor.save_image_on_disk(
                    os.path.join(self._destination, item), image
                )
                results[item] = None
                self.logger.info(f"No poles detected on image {item}")
                continue

            # Slice out the pole(s) detected on the image and run through the
            # angle calculator
            poles = self._slice_out_poles(image, predictions)
            angle_edge_pairs = [
                self._tilt_detector.calculate_inclination(pole)
                for pole in poles
            ]
            results[item] = [str(pair[0]) for pair in angle_edge_pairs]
            self._result_processor.draw_boxes(image, predictions)
            self._result_processor.save_image_on_disk(
                os.path.join(self._destination, item), image
            )
            self.logger.info(f"Image {item} processed")

        if self._to_json:
            self._result_processor.save_as_json(
                os.path.join(self._destination, "out.json"), results
            )
            self.logger.info("Results saved to json")

    def _slice_out_poles(
        self, image: np.ndarray, predictions: t.List[list]
    ) -> t.List[np.ndarray]:
        poles = []
        for prediction in predictions:
            left, top, right, bot, *_ = prediction
            poles.append(image[top:bot, left:right, :])
        return poles
