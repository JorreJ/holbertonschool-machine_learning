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

            x_center = (activated_xy[:, :, :, 0:1] + cx) / output.shape[1]
            y_center = (activated_xy[:, :, :, 1:2] + cy) / output.shape[0]

            input_w = self.model.input_shape[2]
            input_h = self.model.input_shape[1]

            w = (np.exp(t_coords[:, :, :, 2:3]) *
                 self.anchors[i, :, 0].reshape((1, 1, -1, 1))) / input_w
            h = (np.exp(t_coords[:, :, :, 3:4]) *
                 self.anchors[i, :, 1].reshape((1, 1, -1, 1))) / input_h

            scaled_x = x_center * image_size[1]
            scaled_y = y_center * image_size[0]
            scaled_w = w * image_size[1]
            scaled_h = h * image_size[0]

            x1 = scaled_x - (scaled_w / 2)
            y1 = scaled_y - (scaled_h / 2)
            x2 = scaled_x + (scaled_w / 2)
            y2 = scaled_y + (scaled_h / 2)

            boxes.append(np.concatenate((x1, y1, x2, y2), axis=-1))

        return boxes, box_confidences, box_class_probs

    def _sigmoid(self, x):
        """Compute the sigmoid function for a given input x.

        Args:
            x (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The element-wise sigmoid of x.
        """
        return 1 / (1 + np.exp(-x))
