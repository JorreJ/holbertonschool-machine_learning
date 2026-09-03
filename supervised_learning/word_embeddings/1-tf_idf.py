#!/usr/bin/env python3
"""Module that calculates the TF-IDF matrix from a list of sentences."""

import numpy as np

bag_of_words = __import__('0-bag_of_words').bag_of_words


def tf_idf(sentences, vocab=None):
    """Calculate the TF-IDF matrix for a list of sentences.

    Args:
        sentences (list): List of sentences (strings) to analyze.
        vocab (list or numpy.ndarray, optional): List of vocabulary words to
            use. If None, the vocabulary is constructed from all unique words
            in sentences. Defaults to None.

    Returns:
        tuple:
            - embeddings (numpy.ndarray): Matrix of shape (s, f) containing
              the TF-IDF scores for each sentence, where:
                - s is the number of sentences.
                - f is the number of features (vocabulary words).
            - features (numpy.ndarray or list): The vocabulary words used for
              the embeddings, in the same order as the columns.
    """
    bow, features = bag_of_words(sentences, vocab)
    word_counts = bow.sum(axis=1, keepdims=True)
    tf = np.divide(
        bow,
        word_counts,
        out=np.zeros_like(bow, dtype=float),
        where=word_counts != 0
    )
    df = np.sum(bow > 0, axis=0)
    idf = np.log((1 + len(sentences)) / (1 + df)) + 1
    embeddings = tf * idf
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings),
        where=norms != 0
    )
    return embeddings, features
