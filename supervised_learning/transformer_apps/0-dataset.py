#!/usr/bin/env python3
"""Loads and tokenizes a Portuguese-to-English translation dataset."""

from setup import load_pt2en
import transformers


class Dataset:
    """Represent a Portuguese-to-English dataset loader and tokenizer."""

    def __init__(self):
        """Initialize Dataset with training/validation data and tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = (
            self.tokenize_dataset(self.data_train)
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
