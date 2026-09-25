#!/usr/bin/env python3
"""Module that provides a command-line interface for interactive QA loop."""

while True:
    question = input('Q: ')

    if question.lower() in ('exit', 'quit', 'goodbye', 'bye'):
        print('A: Goodbye')
        break

    print('A:')
