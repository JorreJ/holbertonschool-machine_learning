#!/usr/bin/env python3
"""Module that creates a bag-of-words matrix from a list of sentences."""

import string

import numpy as np


def bag_of_words(sentences, vocab=None):
    """Create a bag-of-words matrix.

    Args:
        sentences (list): List of sentences (strings) to analyze.
        vocab (list, optional): List of vocabulary words to use.
            If None, the vocabulary is constructed from all unique words in
            sentences (sorted alphabetically). Defaults to None.

    Returns:
        tuple:
            - embeddings (numpy.ndarray): Matrix of shape (s, f) containing the
              word counts for each sentence, where:
                - s is the number of sentences.
                - f is the number of features (vocabulary words).
            - features (numpy.ndarray or list): The vocabulary words used for
              the embeddings, in the same order as the columns.
    """
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = sentence.replace("'s", "")
        sentence = sentence.translate(
            str.maketrans("", "", string.punctuation))
        clean_sentences.append(sentence)

    words = [sentence.split() for sentence in clean_sentences]

    features = vocab
    if features is None:
        features = np.unique(np.concatenate(words))

    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    feature_to_index = {word: i for i, word in enumerate(features)}

    for i, row in enumerate(words):
        for word in row:
            if word in feature_to_index:
                j = feature_to_index[word]
                embeddings[i, j] += 1

    return embeddings, features
