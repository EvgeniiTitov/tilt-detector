import os
import typing as t

import cv2
import numpy as np
import torch

from helpers import LoggerMixin
from object_detector.abstract_detector import AbstractDetector
from object_detector.yolov4.utils.darknet2pytorch import Darknet
from object_detector.yolov4.utils.utils import nms_cpu


class YoloV4Detector(AbstractDetector, LoggerMixin):
    dependencies_loc = os.path.join(os.path.dirname(__file__), "dependencies")

    def __init__(
        self,
        model_name: str,
        path_to_config: t.Optional[str] = None,
        path_to_weights: t.Optional[str] = None,
        path_to_txt: t.Optional[str] = None,
        confidence: float = 0.25,
        nms: float = 0.3,
        device: str = "gpu",
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
        self._conf_thresh = confidence
        self._nms_thresh = nms
        self._classes = YoloV4Detector.load_classes(self._path_to_txt)
        self.logger.info(
            f"Detecting the following classes: {' '.join(self._classes)}"
        )
        self.logger.info(f"Model {model_name} initialized")

    def process_batch(self, batch_images: t.List[np.ndarray]) -> t.List[list]:
        # Preprocess data: resize, normalization, batch etc
        batch = self._preprocess_batch(batch_images)
        batch = torch.autograd.Variable(batch)
        # Run data through the net
        with torch.no_grad():
            output = self._model(batch)
        # Postprocess data: NMS, thresholding
        boxes = self._postprocess_detections(output)
        boxes = self._rescale_boxes(boxes, batch_images)
        return boxes

    def _preprocess_batch(self, images: t.List[np.ndarray]) -> torch.Tensor:
        processed_images = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self._model.width, self._model.height))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            image = image.div(255).unsqueeze(0)
            processed_images.append(image)
        batch_images = torch.cat(processed_images)
        batch_images = batch_images.to(device=self._device)
        return batch_images

    def _postprocess_detections(self, predictions):
        # [batch, num, 1, 4]
        box_array = predictions[0]
        # [batch, num, num_classes]
        confs = predictions[1]
        if type(box_array).__name__ != "ndarray":
            box_array = box_array.cpu().detach().numpy()
            confs = confs.cpu().detach().numpy()

        num_classes = confs.shape[2]
        # [batch, num, 4]
        box_array = box_array[:, :, 0]
        # [batch, num, num_classes] --> [batch, num]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)
        bboxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = max_conf[i] > self._conf_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]
            bboxes = []
            # nms for each class
            for j in range(num_classes):
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]
                keep = nms_cpu(ll_box_array, ll_max_conf, self._nms_thresh)
                if keep.size > 0:
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]
                    for k in range(ll_box_array.shape[0]):
                        bboxes.append(
                            [
                                ll_box_array[k, 0],
                                ll_box_array[k, 1],
                                ll_box_array[k, 2],
                                ll_box_array[k, 3],
                                ll_max_conf[k],
                                ll_max_conf[k],
                                ll_max_id[k],
                            ]
                        )
            bboxes_batch.append(bboxes)
        return bboxes_batch

    def _rescale_boxes(
        self, boxes_for_batch: list, original_images: t.List[np.ndarray]
    ) -> list:
        if len(boxes_for_batch) != len(original_images):
            raise Exception("Nb of images != nb of detections made by the net")
        boxes_batch_rescaled = []
        for boxes, image in zip(boxes_for_batch, original_images):
            boxes_rescaled = list()
            orig_h, orig_w = image.shape[:2]
            for box in boxes:
                new_left = int(box[0] * orig_w)
                new_left = 2 if new_left <= 0 else new_left
                new_top = int(box[1] * orig_h)
                new_top = 2 if new_top <= 0 else new_top
                new_right = int(box[2] * orig_w)
                new_right = orig_w - 2 if new_right >= orig_w else new_right
                new_bot = int(box[3] * orig_h)
                new_bot = orig_h - 2 if new_bot >= orig_h else new_bot
                if new_left > new_right or new_top > new_bot:
                    self.logger.warning(
                        "Wrong BB coordinates. Expected: left < right, "
                        "actual: %d < %d;   top < bot, actual: %d < %d",
                        new_left,
                        new_right,
                        new_top,
                        new_bot,
                    )
                obj_score = round(box[4], 4)
                conf = round(box[5], 4)
                i = self._classes[box[6]]
                boxes_rescaled.append(
                    [new_left, new_top, new_right, new_bot, obj_score, conf, i]
                )
            boxes_batch_rescaled.append(boxes_rescaled)
        return boxes_batch_rescaled

    @staticmethod
    def load_classes(path_to_txt: str) -> t.List[str]:
        with open(path_to_txt, "r") as file:
            return [item.strip() for item in file.readlines()]
