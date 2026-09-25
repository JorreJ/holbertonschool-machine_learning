#!/usr/bin/env python3
"""Module that performs Question Answering using BERT and TensorFlow Hub."""

from transformers import BertTokenizer
import tensorflow as tf
import tensorflow_hub as hub


def question_answer(question, reference):
    """Find the answer to a question within a reference text using BERT.

    Args:
        question (str): The question to be answered.
        reference (str): The reference text containing the answer.

    Returns:
        str or None: A string containing the answer extracted from the
        reference text, or None if no answer is found.
    """
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    inputs = tokenizer(
        question,
        reference,
        return_tensors="tf",
        truncation=True
    )
    input_ids = inputs["input_ids"]
    input_mask = inputs["attention_mask"]
    segment_ids = inputs["token_type_ids"]
    outputs = model([input_ids, input_mask, segment_ids])
    start_logits = outputs[0]
    end_logits = outputs[1]
    start = tf.argmax(start_logits[0][1:], axis=-1) + 1
    end = tf.argmax(end_logits[0][1:], axis=-1) + 1
    if start > end or start == 0 or end == 0:
        return None
    answer_ids = input_ids[0, start:end + 1]
    text = tokenizer.decode(answer_ids).strip()
    if not text:
        return None
    return text
