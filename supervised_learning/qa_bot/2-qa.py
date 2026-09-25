#!/usr/bin/env python3
"""Module that provides an interactive loop for question-answering."""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """Start an interactive loop to answer questions based on a reference text.

    Args:
        reference (str): The reference text containing the information used to
            answer questions.
    """
    while True:
        question = input('Q: ')

        if question.lower() in ('exit', 'quit', 'goodbye', 'bye'):
            print('A: Goodbye')
            break

        answer = question_answer(question, reference)

        if not answer:
            answer = "Sorry, I do not understand your question."

        print('A:', answer)
