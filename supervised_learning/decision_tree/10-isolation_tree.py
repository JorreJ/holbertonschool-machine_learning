#!/usr/bin/env python3

"""Module for Isolation Random Tree implementation."""

import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """A class representing an Isolation Random Tree for anomaly detection.

    Attributes:
        rng (numpy.random.Generator): Random number generator.
        root (Node): The root node of the tree.
        explanatory (numpy.ndarray): The input features.
        max_depth (int): Maximum depth of the tree.
        predict: Placeholder for prediction logic.
        min_pop (int): Minimum population required to continue splitting.
    """

    def __init__(self, max_depth=10, seed=0, root=None):
        """Initialize a new Isolation Random Tree instance."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Return the string representation of the tree from the root.

        Returns:
            str: Visual representation of the full tree.
        """
        return self.root.__str__() + "\n"

    def depth(self):
        """Get the total depth of the tree.

        Returns:
            int: The maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the nodes in the entire tree.

        Args:
            only_leaves (bool): If True, only count leaf nodes.
                If False, count all nodes in the tree.

        Returns:
            int: The total count of nodes or leaves in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Initialize the bound update starting from the root node."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Gather all leaf nodes in the tree.

        Returns:
            list: A list of all Leaf instances in the tree.
        """
        return self.root.get_leaves_below()

    def update_predict(self):
        """Compute bounds and indicators to build the global predict function.

        This method updates the `predict` attribute with a lambda function
        capable of making batch predictions on multiple samples.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: sum([leaf.indicator(A) * leaf.value
                                      for leaf in leaves])

    def np_extrema(self, arr):
        """Calculate the minimum and maximum of an array.

        Args:
            arr (numpy.ndarray): The input array.

        Returns:
            tuple: (min, max)
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Determine a split using a random feature and threshold.

        Args:
            node (Node): The node to split.

        Returns:
            tuple: (feature_index, threshold_value)
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            pop = node.sub_population
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][pop])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create a leaf child node for isolation.

        Args:
            node (Node): Parent node.
            sub_population (numpy.ndarray): Boolean mask for the child data.

        Returns:
            Leaf: The newly created leaf with depth as value.
        """
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create an internal child node.

        Args:
            node (Node): Parent node.
            sub_population (numpy.ndarray): Boolean mask for the child data.

        Returns:
            Node: The newly created internal node.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively grow the tree by fitting the current node.

        Args:
            node (Node): The node to fit.
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = (node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold))
        right_population = (node.sub_population & (
            self.explanatory[:, node.feature] <= node.threshold))

        # Is left node a leaf ?
        is_left_leaf = ((node.depth == self.max_depth - 1) or
                        (np.sum(left_population) <= self.min_pop))

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = ((node.depth == self.max_depth - 1) or
                         (np.sum(right_population) <= self.min_pop))

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Train the isolation tree on the provided data.

        Args:
            explanatory (numpy.ndarray): Training features.
            verbose (int): If 1, print training summary.
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
