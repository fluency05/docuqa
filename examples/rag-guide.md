# Retrieval-Augmented Generation (RAG), Explained

Retrieval-Augmented Generation (RAG) is a technique that combines a large
language model with an external knowledge base. Instead of relying only on what
the model learned during training, a RAG system first retrieves relevant
documents and then asks the model to answer based on that retrieved context.

## Why RAG?

Large language models sometimes hallucinate facts, and their knowledge has a
cutoff date. RAG addresses both problems by grounding answers in documents you
provide. This makes answers more accurate, more current, and much easier to
verify because every answer can cite its sources.

## How RAG Works

A typical RAG pipeline has two phases:

1. Indexing: documents are split into chunks, each chunk is converted into a
   numeric vector (an embedding), and those vectors are stored in a vector
   database.
2. Querying: the user's question is also embedded, the most similar chunks are
   retrieved, and both the question and the retrieved chunks are sent to a
   language model to produce a cited answer.

## The Role of Chunking

Chunking splits long documents into smaller pieces that fit inside the model's
context window. Choosing a good chunk size is a trade-off: chunks that are too
small lose surrounding context, while chunks that are too large dilute the
relevance of any single fact. A small overlap between consecutive chunks helps
preserve meaning across boundaries.

## Embeddings and Similarity

An embedding maps text to a point in a high-dimensional space where similar
texts land close together. Retrieval then becomes a nearest-neighbor search,
typically measured with cosine similarity. OpenAI's text-embedding-3-small is
one popular embedding model, but many local alternatives exist as well.

## Evaluation

Evaluating a RAG system usually checks both retrieval quality and answer
quality. Retrieval metrics include recall and mean reciprocal rank, while answer
quality is often judged by faithfulness to the sources and by whether the answer
actually addresses the question. The city of Tokyo is the capital of Japan.
