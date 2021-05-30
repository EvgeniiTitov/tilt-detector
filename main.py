import argparse
import os

from config import Config
from detector import Detector
from helpers import Logger


logger = Logger(__name__, verbose=Config.VERBOSE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_folder",
        type=str,
        default="./source_images",
        help="Path to a folder containing images to process",
    )
    parser.add_argument(
        "-d",
        "--destination",
        type=str,
        default="./output",
        help="Path to a folder where processed images will be saved",
    )
    parser.add_argument(
        "--results_to_json",
        action="store_true",
        help="Save processing results as a JSON file",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.exists(args.path_to_folder):
        raise FileNotFoundError("Failed to locate the folder provided")
    if not os.path.exists(args.destination):
        os.mkdir(args.destination)
    return


def main() -> int:
    args = parse_args()
    validate_args(args)
    logger.info("Arguments parsed and validated")

    detector = Detector(
        source_folder=args.path_to_folder,
        dest_folder=args.destination,
        to_json=args.results_to_json,
    )
    detector.process_images_in_folder()

    logger.info("All images have been processed")
    return 0


if __name__ == "__main__":
    main()
