#!/usr/bin/env python3
"""Module that performs semantic search on documents using USE."""

from pathlib import Path
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """Perform a semantic search to find the most relevant document.

    Args:
        corpus_path (str or Path): Path to the directory containing Markdown
            files to search through.
        sentence (str): The input query or sentence to match against the
            documents.

    Returns:
        str: Content of the document that is semantically most similar to the
        input sentence.
    """
    model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    )
    emb_question = np.array(model([sentence])).squeeze()
    files = []
    for file in Path(corpus_path).glob("*.md"):
        files.append(file.read_text(encoding="utf-8"))
    emb_files = np.array(model(files))
    similarities = (
        emb_files @ emb_question / (
            np.linalg.norm(emb_files, axis=1) * np.linalg.norm(emb_question)
        )
    )
    index = similarities.argmax()
    return files[index]
