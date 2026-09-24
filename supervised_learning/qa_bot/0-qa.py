#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
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
    segment_ids = inputs["token_type_ids"]
    input_mask = inputs["attention_mask"]
    outputs = model([input_ids, input_mask, segment_ids])
    start_logits = outputs[0]
    end_logits = outputs[1]
    start = tf.argmax(start_logits, axis=1)[0]
    end = tf.argmax(end_logits, axis=1)[0]
    if start > end:
        return None
    input_ids = input_ids[0, start:end + 1]
    text = tokenizer.decode(input_ids).strip()
    if not text:
        return None
    return text
