#!/usr/bin/env python3
"""Module that calculates the unigram BLEU score for a sentence."""

import numpy as np


def uni_bleu(references, sentence):
    """Calculate the unigram BLEU score for a sentence.

    Args:
        references (list of list of str): List of reference translations,
            where each reference is a list of words.
        sentence (list of str): List containing the model-proposed sentence.

    Returns:
        float: The unigram BLEU score.
    """
    sentence_len = len(sentence)

    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: abs(ref_len - sentence_len)
    )
    count = 0
    for word in set(sentence):
        max_count = max(ref.count(word) for ref in references)
        count += min(sentence.count(word), max_count)
    precision = count / sentence_len
    if sentence_len > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / sentence_len)
    return brevity_penalty * precision
