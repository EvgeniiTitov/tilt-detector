import os

import cv2

from config import Config
from helpers import Logger
from object_detector import YoloV4Detector


logger = Logger(__name__, verbose=Config.VERBOSE)


def main() -> int:
    model = YoloV4Detector("object_detector")

    source = "test_images"
    for item in os.scandir(source):
        if item.is_dir() or not any(
            item.name.lower().endswith(ext) for ext in Config.ALLOWED_EXTS
        ):
            continue
        image = cv2.imread(item.path)
        if image is None:
            logger.error(f"Failed to open image: {item.name}")
            continue
        predictions = model.process_batch([image])[0]

        for prediction in predictions:
            left, top, right, bot, _, conf, cls = prediction  # type: ignore
            cv2.rectangle(image, (left, top), (right, bot), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{cls}_{conf}",
                (left, top + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                2,
            )
        cv2.imshow("", image)
        cv2.waitKey(0)

    return 0


if __name__ == "__main__":
    main()
