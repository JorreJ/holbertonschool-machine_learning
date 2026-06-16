#!/usr/bin/env python3
"""Module that defines the Yolo class for object detection."""

import tensorflow.keras as K


class Yolo:
    """Yolo class to perform object detection using the YOLOv3 algorithm."""

    def __init__(
        self,
        model_path,
        classes_path,
        class_t,
        nms_t,
        anchors
    ):
        """Initialize the Yolo instance.

        Args:
            model_path (str): The path to where a Darknet Keras model is saved.
            classes_path (str): The path to where the list of class names
                used for the model is located, listed in order.
            class_t (float): The box score threshold for the initial
                filtering step.
            nms_t (float): The IOU threshold for non-max suppression.
            anchors (numpy.ndarray): A numpy.ndarray of shape (outputs,
                anchor_boxes, 2) containing all of the anchor boxes.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
