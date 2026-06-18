#!/usr/bin/env python3
"""Module that defines the Yolo class for object detection."""

import tensorflow.keras as K
import numpy as np


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

    def process_outputs(self, outputs, image_size):
        """Process the predictions outputted by the Darknet Keras model.

        Args:
            outputs (list): A list of numpy.ndarrays containing the predictions
                from the model for a single image. Each array has the shape
                (grid_height, grid_width, anchor_boxes, 4 + 1 + classes).
            image_size (numpy.ndarray): A numpy.ndarray containing the image's
                original size [image_height, image_width].

        Returns:
            tuple: A tuple containing:
                - boxes (list): A list of numpy.ndarrays of shape
                  (grid_height, grid_width, anchor_boxes, 4) containing the
                  boundary boxes for each output, scaled to the original image.
                - box_confidences (list): A list of numpy.ndarrays of shape
                  (grid_height, grid_width, anchor_boxes, 1) containing the box
                  confidences for each output.
                - box_class_probs (list): A list of numpy.ndarrays of shape
                  (grid_height, grid_width, anchor_boxes, classes) containing
                  the box class probabilities for each output.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            t_coords = output[:, :, :, :4]
            box_confidence = output[:, :, :, 4:5]
            classes_prob = output[:, :, :, 5:]

            activated_xy = self._sigmoid(t_coords[:, :, :, :2])
            box_confidences.append(self._sigmoid(box_confidence))
            box_class_probs.append(self._sigmoid(classes_prob))

            cx, cy = np.meshgrid(np.arange(output.shape[1]),
                                 np.arange(output.shape[0]))
            cx = cx.reshape(output.shape[0], output.shape[1], 1, 1)
            cy = cy.reshape(output.shape[0], output.shape[1], 1, 1)

            norm_xy = [
                (activated_xy[:, :, :, :1] + cx) / output.shape[1],
                (activated_xy[:, :, :, 1:2] + cy) / output.shape[0]
            ]

            norm_wh = (np.exp(t_coords[:, :, :, 2:]) * self.anchors[i]) / \
                image_size[::-1].reshape(1, 1, 1, 2)

            x_center, y_center = norm_xy[0], norm_xy[1]
            w, h = norm_wh[:, :, :, :1], norm_wh[:, :, :, 1:]
            x1, x2 = x_center - (w / 2), x_center + (w / 2)
            y1, y2 = y_center - (h / 2), y_center + (h / 2)
            sized_x1 = x1 * image_size[1]
            sized_x2 = x2 * image_size[1]
            sized_y1 = y1 * image_size[0]
            sized_y2 = y2 * image_size[0]
            boxes.append(np.concatenate((sized_x1, sized_y1,
                                         sized_x2, sized_y2), axis=-1))

        return boxes, box_confidences, box_class_probs

    def _sigmoid(self, x):
        """Compute the sigmoid function for a given input x.

        Args:
            x (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The element-wise sigmoid of x.
        """
        return 1 / (1 + np.exp(-x))
