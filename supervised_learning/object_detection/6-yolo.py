#!/usr/bin/env python3
"""Module that defines the Yolo class for object detection."""

import os
import cv2
import numpy as np
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

        input_w = self.model.inputs[0].shape[1]
        input_h = self.model.inputs[0].shape[2]

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

    def preprocess_images(self, images):
        """Preprocess a list of images for the Darknet Keras model.

        Resizes images to the model's input size using cubic interpolation,
        rescales pixel values to [0, 1], and records original shapes.

        Args:
            images (list): A list of numpy.ndarrays containing images to
                preprocess.

        Returns:
            tuple: A tuple containing:
                - pimages (numpy.ndarray): Array of shape (N, input_h, input_w,
                  3) containing all preprocessed images, where N is the number
                  of images.
                - image_shapes (numpy.ndarray): Array of shape (N, 2)
                  containing the original shapes [height, width] of each image.
        """
        pimages = []
        image_shapes = []

        input_w = self.model.inputs[0].shape[1]
        input_h = self.model.inputs[0].shape[2]

        for x in images:
            h, w, _ = x.shape
            image_shapes.append([h, w])

            rescaled = cv2.resize(
                x, (input_w, input_h), interpolation=cv2.INTER_CUBIC
            ) / 255.0
            pimages.append(rescaled)

        return np.array(pimages), np.array(image_shapes)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boundary boxes according to box scores and class threshold.

        Args:
            boxes (list): A list of numpy.ndarrays of shape (grid_height,
                grid_width, anchor_boxes, 4) containing the boundary boxes
                for each output.
            box_confidences (list): A list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 1) containing the box
                confidences for each output.
            box_class_probs (list): A list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, classes) containing
                the box class probabilities for each output.

        Returns:
            tuple: A tuple containing:
                - filtered_boxes (list): A list of numpy.ndarrays of shape
                  (?, 4) containing all anchored boxes that pass the threshold.
                - box_classes (list): A list of numpy.ndarrays of shape
                  (?,) containing the class index for each filtered box.
                - box_scores (list): A list of numpy.ndarrays of shape
                  (?,) containing the box score for each filtered box.
        """
        max_boxes = []
        classes = []
        scores = []

        for i in range(len(boxes)):
            box_score = box_confidences[i] * box_class_probs[i]
            class_max = np.argmax(box_score, axis=-1)
            score_max = np.max(box_score, axis=-1)
            mask = score_max >= self.class_t

            max_boxes.append(boxes[i][mask])
            classes.append(class_max[mask])
            scores.append(score_max[mask])

        filtered_boxes = np.concatenate(max_boxes, axis=0)
        box_classes = np.concatenate(classes, axis=0)
        box_scores = np.concatenate(scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Perform Non-max Suppression (NMS) on the filtered boundary boxes.

        Args:
            filtered_boxes (numpy.ndarray): Array of shape (?, 4) containing
                all filtered boundary boxes.
            box_classes (numpy.ndarray): Array of shape (?,) containing the
                class index for each filtered box.
            box_scores (numpy.ndarray): Array of shape (?,) containing the
                box score for each filtered box.

        Returns:
            tuple: A tuple containing:
                - box_predictions (numpy.ndarray): Array of shape (predicted,
                  4) containing the predicted boundary boxes.
                - predicted_box_classes (numpy.ndarray): Array of shape
                  (predicted,) containing the class index for each box.
                - predicted_box_scores (numpy.ndarray): Array of shape
                  (predicted,) containing the box score for each box.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_class = np.unique(box_classes)

        for cls in unique_class:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            order = cls_scores.argsort()[::-1]

            indices = []
            while order.size > 0:
                indices.append(order[0])

                if order.size == 1:
                    break

                ious = self._compute_iou(cls_boxes[order[0]],
                                         cls_boxes[order[1:]])

                filtered_indices = np.where(ious < self.nms_t)[0]

                order = order[filtered_indices + 1]

            for idx in indices:
                box_predictions.append(cls_boxes[idx])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[idx])

        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Display the image with drawn boundary boxes, classes, and scores.

        Draws blue boundary boxes and red labels. If the 's' key is pressed,
        saves the image to a folder named 'detections'.

        Args:
            image (numpy.ndarray): A numpy.ndarray containing an unfiltered
                image.
            boxes (numpy.ndarray): A numpy.ndarray containing the boundary box
                coordinates for each filtering prediction.
            box_classes (numpy.ndarray): A numpy.ndarray containing the class
                index for each box.
            box_scores (numpy.ndarray): A numpy.ndarray containing the box
                score for each box.
            file_name (str): The file path where the original image was loaded
                from (used as the window name and saving file name).
        """
        image_copy = image.copy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]

            image_copy = cv2.rectangle(image, (int(x1), int(y1)),
                                       (int(x2), int(y2)), (255, 0, 0), 2)
            image_copy = cv2.putText(
                image,
                f"{self.class_names[box_classes[i]]} {box_scores[i]:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(file_name, image_copy)
            key = cv2.waitKey(0)

            if key & 0xFF == ord('s'):
                os.makedirs("detections", exist_ok=True)
                save_path = os.path.join("detections", file_name)
                cv2.imwrite(save_path, image_copy)
            cv2.destroyAllWindows()

    def _sigmoid(self, x):
        """Compute the sigmoid function for a given input x.

        Args:
            x (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The element-wise sigmoid of x.
        """
        return 1 / (1 + np.exp(-x))

    def _compute_iou(self, box, boxes):
        """Compute the Intersection over Union (IoU) between box and boxes.

        Args:
            box (numpy.ndarray): Array of shape (4,) containing the boundary
                box coordinates [x1, y1, x2, y2].
            boxes (numpy.ndarray): Array of shape (?, 4) containing multiple
                boundary box coordinates to compare against.

        Returns:
            numpy.ndarray: Array of shape (?,) containing the IoU scores.
        """
        x1, y1, x2, y2 = box
        all_x1 = boxes[:, 0]
        all_y1 = boxes[:, 1]
        all_x2 = boxes[:, 2]
        all_y2 = boxes[:, 3]

        interx1 = np.maximum(x1, all_x1)
        intery1 = np.maximum(y1, all_y1)
        interx2 = np.minimum(x2, all_x2)
        intery2 = np.minimum(y2, all_y2)

        inter_area = np.maximum(0, interx2 - interx1) * \
            np.maximum(0, intery2 - intery1)

        box_area = (x2 - x1) * (y2 - y1)
        boxes_area = (all_x2 - all_x1) * (all_y2 - all_y1)

        union_area = box_area + boxes_area - inter_area

        return inter_area / union_area

    @staticmethod
    def load_images(folder_path):
        """Load all images from a specific folder.

        Args:
            folder_path (str): The path to the folder holding all the
                images to load.

        Returns:
            tuple: A tuple containing:
                - images (list): A list of images as numpy.ndarrays.
                - image_paths (list): A list of paths to the individual
                  images in images.
        """
        images = []
        image_paths = []
        for image in os.listdir(folder_path):
            path = os.path.join(folder_path, image)

            if os.path.isfile(path):
                img = cv2.imread(path)
                if img is not None:
                    image_paths.append(path)
                    images.append(img)

        return images, image_paths
