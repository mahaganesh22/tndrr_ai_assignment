# RAG application

This directory contains a modular Python application for the RAG workflow in `RAG.ipynb`. It uses a Sentence Transformers embedding model, NumPy, and FAISS for retrieval, plus an optional local Transformers model for answer generation. It does not use LangChain or LlamaIndex.

## Layout

```text
rag_app/
  application.py   orchestration and index caching
  chunking.py      text cleanup and token-based chunks
  cli.py           evaluate and ask commands
  config.py        paths and model settings
  document.py      PDF text extraction
  evaluation.py    question loading and Recall@5
  generation.py    optional FLAN-T5 answer generation
  vector_store.py  embeddings, FAISS index, retrieval
requirements.txt
```

## Install and run

From this directory, install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run both chunk-size evaluations and save CSV reports under `outputs/`:

```powershell
python -m rag_app evaluate
```

Ask a question with 256-token chunks:

```powershell
python -m rag_app ask --chunk-size 256 --question "What are the four core functions of the NIST AI RMF?"
```

The embedding model and optional answer model are downloaded by Hugging Face on first use. The application includes the NIST PDF and authored test-question CSV. You can provide alternate files with `--pdf` and `--questions`.

The evaluation reports source-page Recall@5: a question is counted as retrieved when its annotated physical PDF page appears in the top five retrieved chunks. Printed page numbers are included in the detailed evaluation rows for reference.
