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

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes or leaves in the subtree.

        Args:
            only_leaves (bool): If True, only count leaf nodes.
                If False, count all nodes (internal and leaves).

        Returns:
            int: The count of nodes or leaves in the subtree.
        """
        childs = (self.left_child.count_nodes_below(only_leaves=only_leaves) +
                  self.right_child.count_nodes_below(only_leaves=only_leaves))
        if only_leaves is False:
            return childs + 1
        elif only_leaves is True:
            return childs

    def left_child_add_prefix(self, text):
        """Add visual prefixes to the string representation of the left child.

        Args:
            text (str): The string representation of the child node.

        Returns:
            str: The formatted string with tree-like branches.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add visual prefixes to the string representation of the right child.

        Args:
            text (str): The string representation of the child node.

        Returns:
            str: The formatted string with tree-like branches.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0]
        for x in lines[1:]:
            new_text += ("\n       " + x)
        return new_text

    def __str__(self):
        """Return a string representation of the node and its children.

        Returns:
            str: A visual tree representation.
        """
        if self.is_root:
            isroot = "root"
        else:
            isroot = "-> node"
        node = f"{isroot} [feature={self.feature}, threshold={self.threshold}]"
        left_child = self.left_child_add_prefix(str(self.left_child))
        right_child = self.right_child_add_prefix(str(self.right_child))
        tree = node + "\n" + left_child + right_child
        return tree

    def get_leaves_below(self):
        """Retrieve all leaf nodes existing below this node.

        Returns:
            list: A list of Leaf objects found in the subtree.
        """
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Recursively update the feature bounds for all nodes in the subtree.

        This method defines the hyper-rectangle (lower and upper bounds)
        that each node covers in the feature space based on its parents'
        splits.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            new_upper = self.upper.copy()
            new_lower = self.lower.copy()
            if child == self.left_child:
                new_lower[self.feature] = self.threshold
            if child == self.right_child:
                new_upper[self.feature] = self.threshold
            child.upper = new_upper
            child.lower = new_lower

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Create an indicator function for this node.

        The indicator function checks if input data points fall within the
        feature space bounds (lower and upper) defined for this node.
        """
        def is_large_enough(x):
            tests = np.array([np.greater(x[:, key], self.lower[key])
                             for key in list(self.lower.keys())])
            return np.all(tests, axis=0)

        def is_small_enough(x):
            tests = np.array([np.less_equal(x[:, key], self.upper[key])
                             for key in list(self.upper.keys())])
            return np.all(tests, axis=0)

        self.indicator = lambda x: np.all(np.array(
            [is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """Predict the value for a single sample by traversing the subtree.

        Args:
            x (numpy.ndarray): The sample features.

        Returns:
            The predicted value from the corresponding leaf.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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

    def count_nodes_below(self, only_leaves=False):
        """Return the count for a leaf node.

        Args:
            only_leaves (bool): Flag to count only leaves.

        Returns:
            int: Always returns 1 as it is a leaf.
        """
        return 1

    def __str__(self):
        """Return a string representation of the leaf.

        Returns:
            str: The leaf value formatted as a string.
        """
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Return this leaf as a list element.

        Returns:
            list: A list containing only this leaf instance.
        """
        return [self]

    def update_bounds_below(self):
        """Terminate the recursive bound update at the leaf level."""
        pass

    def pred(self, x):
        """Return the prediction value for the leaf.

        Args:
            x (numpy.ndarray): The sample features (not used in leaf).

        Returns:
            The leaf value.
        """
        return self.value


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

    def count_nodes(self, only_leaves=False):
        """Count the nodes in the entire tree.

        Args:
            only_leaves (bool): If True, only count leaf nodes.
                If False, count all nodes in the tree.

        Returns:
            int: The total count of nodes or leaves in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return the string representation of the tree from the root.

        Returns:
            str: Visual representation of the full tree.
        """
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """Gather all leaf nodes in the tree.

        Returns:
            list: A list of all Leaf instances in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Initialize the bound update starting from the root node."""
        self.root.update_bounds_below()

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

    def pred(self, x):
        """Make a prediction for a single sample x.

        Args:
            x (numpy.ndarray): The sample features.

        Returns:
            The predicted value.
        """
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """Train the decision tree on the provided data.

        Args:
            explanatory (numpy.ndarray): Training features.
            target (numpy.ndarray): Training labels.
            verbose (int): If 1, print training summary.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }
    - Accuracy on training data : { self.accuracy(
        self.explanatory,self.target)    }""")

    def np_extrema(self, arr):
        """Calculate min and max of an array.

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

    def fit_node(self, node):
        """Recursively grow the tree by fitting the current node.

        Args:
            node (Node): The node to fit.
        """
        node.feature, node.threshold = self.split_criterion(node)

        left_pop = (node.sub_population &
                    (self.explanatory[:, node.feature] > node.threshold))
        right_pop = (node.sub_population &
                     (self.explanatory[:, node.feature] <= node.threshold))

        # Is left node a leaf ?
        is_left_leaf = (node.depth + 1 >= self.max_depth or
                        sum(left_pop) < self.min_pop or
                        len(np.unique(self.target[left_pop])) == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (node.depth + 1 >= self.max_depth or
                         sum(right_pop) < self.min_pop or
                         len(np.unique(self.target[right_pop])) == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Create a leaf child node.

        Args:
            node (Node): Parent node.
            sub_population (numpy.ndarray): Boolean mask for the child data.

        Returns:
            Leaf: The newly created leaf.
        """
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
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

    def accuracy(self, test_explanatory, test_target):
        """Calculate accuracy on test data.

        Args:
            test_explanatory (numpy.ndarray): Input features.
            test_target (numpy.ndarray): Ground truth labels.

        Returns:
            float: Percentage of correct predictions.
        """
        preds = self.predict(test_explanatory)
        return np.sum(np.equal(preds, test_target)) / test_target.size
