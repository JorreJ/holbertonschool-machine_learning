# Question Answering Bot

This project implements a **question-answering bot** using pretrained NLP models from TensorFlow Hub and Hugging Face.

The project progressively builds a system capable of:

- Extracting an answer from a reference text using **BERT**.
- Interacting with the user through a command-line question-answering loop.
- Searching a corpus of Markdown documents using **semantic similarity**.
- Combining semantic search with BERT to automatically find an answer from a collection of documents.

## Learning Objectives

The main concepts covered by this project are:

- Question answering with pretrained language models.
- Tokenization with BERT.
- Extracting answers from a context using start/end logits.
- Sentence embeddings.
- Semantic search.
- Cosine similarity.
- Building an interactive command-line interface.
- Combining information retrieval and question answering.

## Requirements

The project uses Python 3 and the following libraries:

- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Hub](https://www.tensorflow.org/hub)
- [Transformers](https://huggingface.co/docs/transformers/)
- NumPy

The models used by the project are downloaded automatically when they are loaded for the first time.

## Project Files

### `0-qa.py`

Implements the `question_answer()` function.

The function receives:

```python
question_answer(question, reference)
```

where:

- `question` is the user's question.
- `reference` is the text containing the information needed to answer the question.

The function uses the pretrained BERT model:

```text
bert-large-uncased-whole-word-masking-finetuned-squad
```

and the TensorFlow Hub BERT question-answering model.

The input is tokenized and passed to the model. The model produces two sets of logits:

- start logits, indicating where the answer begins;
- end logits, indicating where the answer ends.

The corresponding tokens are then decoded back into text.

If a valid answer cannot be extracted, the function returns `None`.

### `1-loop.py`

Provides a basic interactive command-line loop.

The program repeatedly asks the user for a question:

```text
Q:
```

and displays the answer:

```text
A:
```

The loop terminates when the user enters one of:

```text
exit
quit
goodbye
bye
```

### `2-qa.py`

Extends the interactive loop by connecting it to the BERT question-answering function.

The `answer_loop()` function receives a reference text and repeatedly:

1. Reads a question from the user.
2. Checks whether the user wants to exit.
3. Uses `question_answer()` to find an answer in the reference text.
4. Displays the answer.

If no answer is found, the bot responds:

```text
Sorry, I do not understand your question.
```

### `3-semantic_search.py`

Implements semantic search over a directory containing Markdown (`.md`) files.

The `semantic_search()` function receives:

```python
semantic_search(corpus_path, sentence)
```

It uses the **Universal Sentence Encoder Large** from TensorFlow Hub to convert:

- the user's query;
- every Markdown document in the corpus

into numerical embeddings.

The similarity between the query embedding and each document embedding is then calculated using **cosine similarity**.

The document with the highest similarity score is returned.

Conceptually, the cosine similarity is:

```text
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

This allows the search to find documents that are semantically related to the question, even when they do not contain exactly the same words.

### `4-qa.py`

Combines the previous components into a complete question-answering system.

For every question, the program:

1. Uses semantic search to find the most relevant document.
2. Uses BERT question answering to extract the answer from that document.
3. Displays the answer to the user.

The overall pipeline is therefore:

```text
User question
      │
      ▼
Semantic Search
      │
      ▼
Most relevant Markdown document
      │
      ▼
BERT Question Answering
      │
      ▼
Extracted answer
```

## Usage

From the `qa_bot` directory, the individual programs can be run according to their respective tasks.

For the final question-answering system, provide the path to the directory containing the Markdown documents.

Example:

```bash
./4-qa.py
```

The program starts an interactive session:

```text
Q: When are PLDs?
A: ...
```

To exit:

```text
Q: exit
A: Goodbye
```

The corpus is expected to contain Markdown files (`.md`). The semantic search implementation reads these files and uses their contents to create the document embeddings.

## How It Works

The final system combines two different NLP techniques.

### 1. Semantic Search

The Universal Sentence Encoder converts text into high-dimensional vectors called **embeddings**.

For example:

```text
"When are PLDs?"
        │
        ▼
   Text embedding
        │
        ▼
[0.12, -0.34, ..., 0.57]
```

The same process is applied to every document in the corpus.

The cosine similarity between the question embedding and each document embedding is then calculated. The document with the highest score is selected.

### 2. Question Answering

Once the most relevant document has been identified, the question and document are passed to the BERT question-answering model.

BERT predicts the beginning and end positions of the answer within the reference text.

For example:

```text
Reference:
"PLDs are performed every Friday afternoon."

Question:
"When are PLDs?"

Answer:
"every Friday afternoon"
```

This two-stage approach separates **document retrieval** from **answer extraction**.

## Models

The project uses the following pretrained models.

### BERT

The tokenizer is loaded from:

```text
bert-large-uncased-whole-word-masking-finetuned-squad
```

This model is designed for extractive question answering using the SQuAD dataset.

The question-answering model is loaded from TensorFlow Hub:

```text
https://tfhub.dev/see--/bert-uncased-tf2-qa/1
```

### Universal Sentence Encoder

Semantic search uses:

```text
https://tfhub.dev/google/universal-sentence-encoder-large/5
```

The encoder transforms sentences and documents into embeddings that can be compared using cosine similarity.

## Limitations

This implementation is intentionally simple and is designed for learning purposes.

- Only Markdown files directly inside the specified corpus directory are searched.
- The complete contents of each document are embedded.
- Only the single document with the highest cosine similarity is selected.
- The answer is extracted from that document using BERT.
- The quality of the final answer depends on both the semantic search result and BERT's ability to identify an answer in the selected document.
- The pretrained models can require significant memory and processing time.

## Technologies

- Python 3
- TensorFlow
- TensorFlow Hub
- Hugging Face Transformers
- NumPy
- BERT
- Universal Sentence Encoder
- Natural Language Processing (NLP)
- Semantic Search
- Cosine Similarity
