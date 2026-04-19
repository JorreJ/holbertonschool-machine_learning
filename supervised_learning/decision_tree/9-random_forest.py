#!/usr/bin/env python3
"""Module for Random Forest classifier implementation."""

import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """A class representing a Random Forest classifier.

    Attributes:
        n_trees (int): Number of trees in the forest.
        max_depth (int): Maximum depth allowed for each tree.
        min_pop (int): Minimum population required to split a node.
        seed (int): Seed for random number generation.
        numpy_preds (list): List of predict functions from trained trees.
        target (numpy.ndarray): Training target values.
        explanatory (numpy.ndarray): Training input features.
    """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize a new Random Forest instance."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Predict classes for explanatory variables using majority voting.

        Args:
            explanatory (numpy.ndarray): The input features to predict.

        Returns:
            numpy.ndarray: The predicted classes.
        """
        all_trees_preds = []
        for p in self.numpy_preds:
            all_trees_preds.append(p(explanatory))

        matrix_preds = np.transpose(np.array(all_trees_preds))
        final_pred = []
        for vote in matrix_preds:
            counts = np.bincount(vote)
            winner = np.argmax(counts)
            final_pred.append(winner)
        return np.array(final_pred)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Train the random forest on the provided data.

        Args:
            explanatory (numpy.ndarray): Training features.
            target (numpy.ndarray): Training labels.
            n_trees (int): Number of trees to train.
            verbose (int): If 1, print training statistics.
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop,
                              seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)

            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : { np.array(depths).mean()      }
    - Mean number of nodes           : { np.array(nodes).mean()       }
    - Mean number of leaves          : { np.array(leaves).mean()      }
    - Mean accuracy on training data : { np.array(accuracies).mean()  }
    - Accuracy of the forest on td   : { self.accuracy(
        self.explanatory, self.target) }""")

    def accuracy(self, test_explanatory, test_target):
        """Calculate the accuracy of the forest on a test set.

        Args:
            test_explanatory (numpy.ndarray): Test features.
            test_target (numpy.ndarray): True labels for the test set.

        Returns:
            float: The accuracy score.
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                      test_target)) / test_target.size
