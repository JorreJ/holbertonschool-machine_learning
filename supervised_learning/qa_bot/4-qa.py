#!/usr/bin/env python3
"""Module that provides an interactive QA loop using semantic search."""

answer_search = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """Start an interactive question-answering loop over a document corpus.

    Args:
        corpus_path (str or Path): Path to the directory containing Markdown
            documents to search and retrieve answers from.
    """
    while True:
        question = input('Q: ')

        if question.lower() in ('exit', 'quit', 'goodbye', 'bye'):
            print('A: Goodbye')
            break

        reference = semantic_search(corpus_path, question)

        answer = answer_search(question, reference)

        if not answer:
            answer = "Sorry, I do not understand your question."

        print('A:', answer)
