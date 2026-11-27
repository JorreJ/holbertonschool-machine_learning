#!/usr/bin/env python3
"""Module that creates, builds, and trains a gensim Word2Vec model."""

from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a gensim Word2Vec model.

    Args:
        sentences (list of list of str): List of sentences to be trained on.
        vector_size (int): Dimensionality of the feature layer. Defaults to
            100.
        min_count (int): Minimum number of occurrences of a word for use in
            training. Defaults to 5.
        window (int): Maximum distance between the current and predicted word
            within a sentence. Defaults to 5.
        negative (int): Size of negative sampling. Defaults to 5.
        cbow (bool): Training flag; True is for CBOW, False is for Skip-gram.
            Defaults to True.
        epochs (int): Number of iterations to train over. Defaults to 5.
        seed (int): Seed for the random number generator. Defaults to 0.
        workers (int): Number of worker threads to train the model. Defaults
            to 1.

    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.
    """
    sg = 0 if cbow else 1
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    return model
