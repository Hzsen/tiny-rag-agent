# TODO

## Environment & Infrastructure (PDM, Folder structure)
- [x] Confirm Python 3.12+ is installed and available in shell.
- [x] Install `pdm` and verify with `pdm --version`.
- [x] Run `pdm install -v` to create the venv and install deps.
- [x] Run `pdm run check-installation` to validate setup.
- [x] Copy `.env.example` to `.env` and fill required values.
- [x] Create/verify project folders: `ingestion/`, `retrieval/`, `graph/`, `cli/`.
- [x] Add a `config/` module for settings and model paths.
- [x] Document local model paths for MLX/Qwen/DeepSeek in `.env`.

## Data Ingestion Module (PDF loading, Chunking)
- [x] Define ingestion config schema with Pydantic v2 (paths, chunk size, overlap).
- [x] Implement PDF loader that yields raw text per document.
- [x] Implement Markdown loader for parity with PDFs.
- [x] Add text cleaning/normalization step (whitespace, headers).
- [x] Implement chunking strategy (size + overlap) with metadata.
- [x] Add document model for chunks with `doc_id`, `source`, `page`, `text`.
- [ ] Persist chunk outputs for debugging (optional local cache).
- [ ] Add ingestion unit tests for loader + chunking.

## Retrieval Module (Chroma + BM25 Class)
- [ ] Define retrieval config schema (top_k, weights, vector dir).
- [x] Implement BM25 index class (fit, query, serialize).
- [x] Implement Chroma vector store wrapper (build, query, persist).
- [x] Add embedding model wrapper (MLX or API fallback).
- [x] Implement hybrid retriever that combines BM25 + Chroma scores.
- [ ] Add relevance grading input/output schema for retrieval results.
- [ ] Add tests for hybrid scoring and result ordering.

## Graph State & Nodes (LangGraph setup)
- [x] Define LangGraph state schema (query, docs, scores, answer).
- [x] Implement Retrieve node (hybrid retriever call).
- [x] Implement Grade node (relevance scoring + filter).
- [x] Implement Rewrite node for query reformulation.
- [x] Implement Generate node (LLM response with citations).
- [ ] Implement Check node (hallucination/grounding check).
- [x] Wire edges to match Retrieve -> Grade -> Generate -> Check -> Loop.
- [x] Add termination conditions for grounded final output.
- [ ] Add node-level error handling with specific exceptions.

## Main Workflow & CLI Entrypoint
- [ ] Create CLI entrypoint with `pdm run main --mode local`.
- [ ] Add CLI args for mode, model path, and input file.
- [ ] Load config and initialize retriever + graph in `main`.
- [ ] Add logging for each node transition and final output.
- [ ] Provide sample run command in README or `cli/README.md`.
