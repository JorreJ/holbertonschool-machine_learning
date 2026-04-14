#!/usr/bin/env python3

"""Module for decision tree structure and depth calculations."""

import numpy as np


class Node:
    """A class representing a node in a decision tree.

    Attributes:
        feature (int): The index of the feature used for the split.
        threshold (float): The threshold value for the split.
        left_child (Node): The left child of the node.
        right_child (Node): The right child of the node.
        is_leaf (bool): Whether the node is a leaf.
        is_root (bool): Whether the node is the root of the tree.
        sub_population (numpy.ndarray): The subset of data at this node.
        depth (int): The depth level of the node.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a new Node instance."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculate the maximum depth below this node.

        Returns:
            int: The maximum depth reachable from this node.
        """
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)


class Leaf(Node):
    """A class representing a leaf node in a decision tree.

    Attributes:
        value: The prediction value of the leaf.
        depth (int): The depth level of the leaf.
    """

    def __init__(self, value, depth=None):
        """Initialize a new Leaf instance."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of the leaf.

        Returns:
            int: The depth of this leaf.
        """
        return self.depth


class Decision_Tree():
    """A class representing a Decision Tree classifier or regressor.

    Attributes:
        rng (numpy.random.Generator): Random number generator.
        root (Node): The root node of the tree.
        explanatory (numpy.ndarray): The input features.
        target (numpy.ndarray): The target values.
        max_depth (int): Maximum depth of the tree.
        min_pop (int): Minimum population required to split.
        split_criterion (str): The criterion used for splitting.
        predict: Placeholder for prediction logic.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize the Decision Tree with given parameters."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Get the total depth of the tree.

        Returns:
            int: The maximum depth of the tree.
        """
        return self.root.max_depth_below()
