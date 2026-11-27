#!/usr/bin/env python3
"""Module that implements the Isolation Random Forest anomaly detector."""

import numpy as np


Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest:
    """Represent an Isolation Random Forest for anomaly detection.

    Attributes:
        numpy_predicts (list): List of prediction outputs.
        target (numpy.ndarray or None): Target array, initialized to None.
        numpy_preds (list or None): List of prediction callables for each
            tree in the forest.
        n_trees (int): Number of trees in the forest.
        max_depth (int): Maximum depth allowed for each tree.
        seed (int): Base random seed for reproducibility.
        explanatory (numpy.ndarray): Training dataset features.
    """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize an Isolation_Random_Forest instance.

        Args:
            n_trees (int): Number of isolation trees in the forest. Defaults
                to 100.
            max_depth (int): Maximum depth of each isolation tree. Defaults to
                10.
            min_pop (int): Minimum population size for splitting. Defaults to
                1.
            seed (int): Base random seed. Defaults to 0.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """Predict the mean isolation depth for each sample.

        Args:
            explanatory (numpy.ndarray): Dataset of shape (n_samples,
                n_features) to evaluate.

        Returns:
            numpy.ndarray: Array of mean path depths across all trees.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """Fit the Isolation Random Forest to the explanatory dataset.

        Args:
            explanatory (numpy.ndarray): Dataset of shape (n_samples,
                n_features) used to build the trees.
            n_trees (int): Number of trees to train. Defaults to 100.
            verbose (int): Verbosity flag; prints training statistics when set
                to 1. Defaults to 0.
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(
                max_depth=self.max_depth, seed=self.seed + i
            )
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
- Mean depth                     : {np.array(depths).mean()}
- Mean number of nodes           : {np.array(nodes).mean()}
- Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        """Return the rows in explanatory with the smallest mean depth.

        Args:
            explanatory (numpy.ndarray): Dataset of shape (n_samples,
                n_features) to evaluate.
            n_suspects (int): Number of top anomaly suspects to return.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: Subset of explanatory rows corresponding to
                  the n_suspects smallest mean depths.
                - numpy.ndarray: Corresponding smallest mean depths.
        """
        depths = self.predict(explanatory)
        indices = np.argsort(depths)[:n_suspects]
        return explanatory[indices], depths[indices]
