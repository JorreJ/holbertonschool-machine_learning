#!/usr/bin/env python3
"""Loads and tokenizes a Portuguese-to-English translation dataset."""

from setup import load_pt2en
import tensorflow as tf
import transformers


class Dataset:
    """Represent a Portuguese-to-English dataset loader and tokenizer."""

    def __init__(self, batch_size, max_len):
        """Initialize Dataset with training/validation data and tokenizers.

        Args:
            batch_size (int): The batch size for training and validation data.
            max_len (int): The maximum number of tokens allowed per example.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = (
            self.tokenize_dataset(self.data_train)
        )
        self.data_train = self.data_train.map(self.tf_encode).filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )
        ).cache().shuffle(20000).padded_batch(
            batch_size,
            padded_shapes=([None], [None])
        ).prefetch(tf.data.experimental.AUTOTUNE)
        self.data_valid = self.data_valid.map(self.tf_encode).filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )
        ).padded_batch(
            batch_size,
            padded_shapes=([None], [None])
        )

    def tokenize_dataset(self, data):
        """Train and return tokenizers for Portuguese and English data.

        Args:
            data (tf.data.Dataset): Dataset containing tuples of Portuguese and
                English text tensors.

        Returns:
            tuple:
                - tokenizer_pt (transformers.PreTrainedTokenizerFast):
                  The trained Portuguese tokenizer.
                - tokenizer_en (transformers.PreTrainedTokenizerFast):
                  The trained English tokenizer.
        """
        pretrain_pt_tokenizer = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
        )
        pretrain_en_tokenizer = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )
        pt_text = (pt.numpy().decode('utf-8') for pt, _ in data)
        en_text = (en.numpy().decode('utf-8') for _, en in data)
        tokenizer_pt = pretrain_pt_tokenizer.train_new_from_iterator(
            pt_text,
            vocab_size=2**13
        )
        tokenizer_en = pretrain_en_tokenizer.train_new_from_iterator(
            en_text,
            vocab_size=2**13
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode a pair of Portuguese and English text tensors into tokens.

        Args:
            pt (tf.Tensor): Tensor containing the Portuguese sentence.
            en (tf.Tensor): Tensor containing the English sentence.

        Returns:
            tuple:
                - pt_tokens (list): List of integer tokens representing the
                  Portuguese sentence including start and end tokens.
                - en_tokens (list): List of integer tokens representing the
                  English sentence including start and end tokens.
        """
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size
        pt = pt.numpy().decode('utf-8')
        en = en.numpy().decode('utf-8')
        pt_tokens = self.tokenizer_pt.encode(pt, add_special_tokens=False)
        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = self.tokenizer_en.encode(en, add_special_tokens=False)
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Wrap the encode method as a TensorFlow Python function.

        Args:
            pt (tf.Tensor): Tensor containing the Portuguese sentence.
            en (tf.Tensor): Tensor containing the English sentence.

        Returns:
            tuple:
                - pt_tokens (tf.Tensor): Tensor of integer tokens for
                  the Portuguese sentence (dtype int64).
                - en_tokens (tf.Tensor): Tensor of integer tokens for
                  the English sentence (dtype int64).
        """
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )
        return pt_tokens, en_tokens
