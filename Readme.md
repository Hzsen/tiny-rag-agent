# tiny-rag-agent

A lightweight, agentic Retrieval-Augmented Generation (RAG) system built for technical documentation analysis. This project implements a "Self-Reflective" workflow using LangGraph, combining hybrid search strategies with iterative reasoning to reduce hallucinations.

The environment and dependency management infrastructure is adapted from [tiny-llm](https://github.com/skyzh/tiny-llm).

## System Architecture

The core logic is a state machine that critiques its own retrieval and generation quality before finalizing an answer.

```mermaid
graph TD
    Start([Start]) --> Retrieve[Retrieve Documents]
    Retrieve --> Grade{Grade Relevance}
    Grade -->|Relevant| Generate[Generate Answer]
    Grade -->|Irrelevant| Rewrite[Rewrite Query]
    Rewrite --> Retrieve
    Generate --> Check{Hallucination Check}
    Check -->|Grounded| Final([Final Output])
    Check -->|Hallucination| Generate
    Check -->|Not Answered| Rewrite

```

## Key Features

* **Hybrid Search**: Implements an ensemble retriever combining BM25 (keyword) and ChromaDB (semantic) to improve recall on technical terms.
* **Agentic Control Flow**: Uses LangGraph to implement cyclical reasoning (Retrieve -> Grade -> Generate -> Reflect).
* **Local-First Design**: Optimized for running on Apple Silicon using MLX (via the tiny-llm foundation) or efficient API bridging.
* **Structured Output**: Enforces strict schema adherence for document citations and relevance scoring.

## Environment Setup

This project uses `pdm` for dependency management and requires a Macintosh device with Apple Silicon (M1/M2/M3) for local inference optimizations, following the [tiny-llm](https://github.com/skyzh/tiny-llm) setup guide.

### Prerequisites

1. **Install pdm**: Follow the [official guide](https://www.google.com/search?q=https://pdm-project.org/en/latest/%23installation) to install pdm.
2. **Hugging Face CLI**: Required for downloading model weights.

### Installation

Clone the repository and install dependencies. PDM will automatically create a virtual environment.

```bash
git clone [https://github.com/Hzsen/tiny-rag-agent](https://github.com/Hzsen/tiny-rag-agent)
cd tiny-rag-agent
pdm install -v

```

### Verify Installation

Run the installation check to ensure the environment and dependencies are correctly configured.

```bash
pdm run check-installation

```

### Model Preparation (Qwen2 / DeepSeek)

We use Qwen2-Instruct or DeepSeek models for inference. If running locally via MLX, download the weights using `huggingface-cli`.

**Note:** The 7B model requires approximately 20GB of memory. Use the 0.5B model for testing on devices with lower RAM.

```bash
huggingface-cli login

# For low memory environments
huggingface-cli download Qwen/Qwen2-0.5B-Instruct-MLX

# Recommended for better reasoning
huggingface-cli download Qwen/Qwen2-7B-Instruct-MLX

```

## Usage

### Configuration

Copy the example environment file and configure your API keys (if using cloud fallbacks) or local model paths.

```bash
cp .env.example .env

```

### Running the Agent

To start the RAG pipeline with the default settings:

```bash
pdm run main --mode local

```

## Roadmap

| Component | Status | Description |
| --- | --- | --- |
| **Ingestion** | ✅ | PDF/Markdown loading and chunking |
| **Retrieval** | ✅ | Hybrid Search (BM25 + Vector) |
| **Graph Logic** | ✅ | Basic Re-ranking and Hallucination checks |
| **Serving** | 🚧 | FastAPI integration for external calls |
| **Eval** | 🚧 | RAGAS evaluation pipeline |

## License

MIT