# TODO

## Environment & Infrastructure (PDM, Folder structure)
- [ ] Confirm Python 3.12+ is installed and available in shell.
- [ ] Install `pdm` and verify with `pdm --version`.
- [ ] Run `pdm install -v` to create the venv and install deps.
- [ ] Run `pdm run check-installation` to validate setup.
- [ ] Copy `.env.example` to `.env` and fill required values.
- [ ] Create/verify project folders: `ingestion/`, `retrieval/`, `graph/`, `cli/`.
- [ ] Add a `config/` module for settings and model paths.
- [ ] Document local model paths for MLX/Qwen/DeepSeek in `.env`.

## Data Ingestion Module (PDF loading, Chunking)
- [ ] Define ingestion config schema with Pydantic v2 (paths, chunk size, overlap).
- [ ] Implement PDF loader that yields raw text per document.
- [ ] Implement Markdown loader for parity with PDFs.
- [ ] Add text cleaning/normalization step (whitespace, headers).
- [ ] Implement chunking strategy (size + overlap) with metadata.
- [ ] Add document model for chunks with `doc_id`, `source`, `page`, `text`.
- [ ] Persist chunk outputs for debugging (optional local cache).
- [ ] Add ingestion unit tests for loader + chunking.

## Retrieval Module (Chroma + BM25 Class)
- [ ] Define retrieval config schema (top_k, weights, vector dir).
- [ ] Implement BM25 index class (fit, query, serialize).
- [ ] Implement Chroma vector store wrapper (build, query, persist).
- [ ] Add embedding model wrapper (MLX or API fallback).
- [ ] Implement hybrid retriever that combines BM25 + Chroma scores.
- [ ] Add relevance grading input/output schema for retrieval results.
- [ ] Add tests for hybrid scoring and result ordering.

## Graph State & Nodes (LangGraph setup)
- [ ] Define LangGraph state schema (query, docs, scores, answer).
- [ ] Implement Retrieve node (hybrid retriever call).
- [ ] Implement Grade node (relevance scoring + filter).
- [ ] Implement Rewrite node for query reformulation.
- [ ] Implement Generate node (LLM response with citations).
- [ ] Implement Check node (hallucination/grounding check).
- [ ] Wire edges to match Retrieve -> Grade -> Generate -> Check -> Loop.
- [ ] Add termination conditions for grounded final output.
- [ ] Add node-level error handling with specific exceptions.

## Main Workflow & CLI Entrypoint
- [ ] Create CLI entrypoint with `pdm run main --mode local`.
- [ ] Add CLI args for mode, model path, and input file.
- [ ] Load config and initialize retriever + graph in `main`.
- [ ] Add logging for each node transition and final output.
- [ ] Provide sample run command in README or `cli/README.md`.
