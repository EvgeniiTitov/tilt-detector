import json
import typing as t

import cv2
import numpy as np


class ResultProcessor:
    @staticmethod
    def draw_boxes(image: np.ndarray, predictions: t.List[list]) -> None:
        for prediction in predictions:
            left, top, right, bot, _, conf, cls = prediction  # type: ignore
            cv2.rectangle(image, (left, top), (right, bot), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{cls}_{conf:.4f}",
                (left, top + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )

    @staticmethod
    def save_as_json(filename: str, results: t.Mapping) -> None:
        with open(filename, "w") as json_file:
            json.dump(results, json_file, indent=4)

    @staticmethod
    def save_image_on_disk(filepath: str, image: np.ndarray) -> None:
        cv2.imwrite(filepath, image)
