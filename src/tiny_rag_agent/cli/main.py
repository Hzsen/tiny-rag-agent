"""CLI entrypoint for the tiny-rag-agent workflow."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from openai import OpenAI
from pydantic import BaseModel, Field, PrivateAttr

from tiny_rag_agent.config.settings import Settings
from tiny_rag_agent.graph.workflow import build_workflow
from tiny_rag_agent.ingestion.chunker import Chunker
from tiny_rag_agent.ingestion.schema import DocumentChunk, IngestionConfig
from tiny_rag_agent.retrieval.hybrid import HybridRetriever
from tiny_rag_agent.retrieval.keyword_store import KeywordStore
from tiny_rag_agent.retrieval.vector_store import VectorStore


class LocalHeuristicChatModel(BaseChatModel):
    """Minimal local chat model for CLI bootstrapping."""

    model_name: str = "local-heuristic"

    @property
    def _llm_type(self) -> str:
        return "local-heuristic"

    def _generate(  # type: ignore[override]
        self,
        messages: Sequence[BaseMessage],
        stop: Iterable[str] | None = None,
        run_manager=None,
        **kwargs: object,
    ) -> ChatResult:
        content = _heuristic_response(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def with_structured_output(  # type: ignore[override]
        self,
        schema: dict | type,
        *,
        include_raw: bool = False,
        **kwargs: object,
    ) -> Runnable[object, BaseModel | dict]:
        if kwargs:
            raise ValueError(f"Unsupported arguments: {sorted(kwargs)}")

        def _structured(_: object) -> BaseModel | dict:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                if "score" in schema.model_fields:
                    return schema(score="yes")
                return schema()
            return {"score": "yes"}

        def _structured_with_raw(_: object) -> dict:
            parsed = _structured(_)
            return {
                "raw": AIMessage(content="yes"),
                "parsed": parsed,
                "parsing_error": None,
            }

        return RunnableLambda(_structured_with_raw if include_raw else _structured)


class CloudChatModel(BaseChatModel):
    """OpenAI-backed chat model with JSON-mode structured output."""

    model_name: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.0)
    _client: OpenAI = PrivateAttr()

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._client = OpenAI()

    @property
    def _llm_type(self) -> str:
        return "openai"

    def _generate(  # type: ignore[override]
        self,
        messages: Sequence[BaseMessage],
        stop: Iterable[str] | None = None,
        run_manager=None,
        **kwargs: object,
    ) -> ChatResult:
        content = self._call_openai(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def with_structured_output(  # type: ignore[override]
        self,
        schema: dict | type,
        *,
        include_raw: bool = False,
        **kwargs: object,
    ) -> Runnable[object, BaseModel | dict]:
        if kwargs:
            raise ValueError(f"Unsupported arguments: {sorted(kwargs)}")

        def _structured(input_data: object) -> BaseModel | dict:
            messages = _normalize_messages(input_data)
            schema_prompt = _schema_prompt(schema)
            messages = [SystemMessage(content=schema_prompt), *messages]
            content = self._call_openai(messages, json_mode=True)
            payload = json.loads(content)
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(payload)
            return payload

        def _structured_with_raw(input_data: object) -> dict:
            messages = _normalize_messages(input_data)
            schema_prompt = _schema_prompt(schema)
            messages = [SystemMessage(content=schema_prompt), *messages]
            content = self._call_openai(messages, json_mode=True)
            payload = json.loads(content)
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                parsed = schema.model_validate(payload)
            else:
                parsed = payload
            return {
                "raw": AIMessage(content=content),
                "parsed": parsed,
                "parsing_error": None,
            }

        return RunnableLambda(_structured_with_raw if include_raw else _structured)

    def _call_openai(
        self,
        messages: Sequence[BaseMessage],
        json_mode: bool = False,
    ) -> str:
        openai_messages = [_message_to_openai(message) for message in messages]
        response_format = {"type": "json_object"} if json_mode else None
        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=openai_messages,
            response_format=response_format,
        )
        choice = response.choices[0]
        return (choice.message.content or "").strip()


class DeepseekChatModel(BaseChatModel):
    """DeepSeek-backed chat model via OpenAI-compatible API."""

    model_name: str = Field(default="deepseek-chat")
    temperature: float = Field(default=0.0)
    _client: OpenAI = PrivateAttr()

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Missing required environment variable: DEEPSEEK_API_KEY")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _generate(  # type: ignore[override]
        self,
        messages: Sequence[BaseMessage],
        stop: Iterable[str] | None = None,
        run_manager=None,
        **kwargs: object,
    ) -> ChatResult:
        content = self._call_deepseek(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def with_structured_output(  # type: ignore[override]
        self,
        schema: dict | type,
        *,
        include_raw: bool = False,
        **kwargs: object,
    ) -> Runnable[object, BaseModel | dict]:
        if kwargs:
            raise ValueError(f"Unsupported arguments: {sorted(kwargs)}")

        def _structured(input_data: object) -> BaseModel | dict:
            messages = _normalize_messages(input_data)
            schema_prompt = _schema_prompt(schema)
            messages = [SystemMessage(content=schema_prompt), *messages]
            content = self._call_deepseek(messages, json_mode=True)
            payload = json.loads(content)
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(payload)
            return payload

        def _structured_with_raw(input_data: object) -> dict:
            messages = _normalize_messages(input_data)
            schema_prompt = _schema_prompt(schema)
            messages = [SystemMessage(content=schema_prompt), *messages]
            content = self._call_deepseek(messages, json_mode=True)
            payload = json.loads(content)
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                parsed = schema.model_validate(payload)
            else:
                parsed = payload
            return {
                "raw": AIMessage(content=content),
                "parsed": parsed,
                "parsing_error": None,
            }

        return RunnableLambda(_structured_with_raw if include_raw else _structured)

    def _call_deepseek(
        self,
        messages: Sequence[BaseMessage],
        json_mode: bool = False,
    ) -> str:
        openai_messages = [_message_to_openai(message) for message in messages]
        response_format = {"type": "json_object"} if json_mode else None
        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=openai_messages,
            response_format=response_format,
        )
        choice = response.choices[0]
        return (choice.message.content or "").strip()


def _heuristic_response(messages: Sequence[BaseMessage]) -> str:
    """Return a deterministic response based on prompt content."""
    if not messages:
        return ""

    content = messages[-1].content if hasattr(messages[-1], "content") else ""
    if not isinstance(content, str):
        content = str(content)

    if "Rewrite it for retrieval" in content:
        marker = "Original question:\n"
        if marker in content:
            question = content.split(marker, 1)[-1]
            question = question.split("\n\n", 1)[0]
            return question.strip()
        return content.strip()

    if "Documents:" in content and "Question:" in content:
        docs_marker = "Documents:\n"
        docs = content.split(docs_marker, 1)[-1]
        docs = docs.strip()
        excerpt = docs[:400].strip()
        if not excerpt:
            return "No documents were provided."
        return f"Based on the documents: {excerpt}"

    if "Return only 'yes' or 'no'" in content:
        return "yes"

    return "yes"


def _build_generate_chain(llm: BaseChatModel) -> Runnable:
    """Create a simple generation chain for local runs."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question using the provided documents."),
            (
                "human",
                "Question:\n{question}\n\nDocuments:\n{documents}\n\n"
                "Provide a concise answer grounded in the documents.",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def _load_chunks(input_file: Path | None) -> list[DocumentChunk]:
    """Load and chunk documents from a file or directory."""
    config = IngestionConfig()
    chunker = Chunker(config)

    if input_file is not None:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        return chunker.process_document(str(input_file))
    return []


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tiny-rag-agent CLI")
    parser.add_argument(
        "--mode",
        choices=["local", "cloud", "deepseek"],
        default="local",
        help="Execution mode (local or cloud).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Optional file to ingest (PDF or Markdown).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Question to ask the agent (prompted if omitted).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if vector DB exists.",
    )
    return parser.parse_args(argv)


def _prompt_for_query() -> str:
    query = input("Enter your question: ").strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    return query


def _message_to_openai(message: BaseMessage) -> dict[str, str]:
    role = "user"
    if isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    return {"role": role, "content": str(message.content)}


def _normalize_messages(input_data: object) -> list[BaseMessage]:
    if isinstance(input_data, list) and all(
        isinstance(item, BaseMessage) for item in input_data
    ):
        return list(input_data)
    if isinstance(input_data, BaseMessage):
        return [input_data]
    if isinstance(input_data, str):
        return [HumanMessage(content=input_data)]
    if isinstance(input_data, dict) and "messages" in input_data:
        messages = input_data["messages"]
        if isinstance(messages, list) and all(
            isinstance(item, BaseMessage) for item in messages
        ):
            return list(messages)
    return [HumanMessage(content=str(input_data))]


def _schema_prompt(schema: dict | type) -> str:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        fields = ", ".join(schema.model_fields.keys())
    elif isinstance(schema, dict):
        fields = ", ".join(schema.keys())
    else:
        fields = "score"
    return (
        "Return a valid JSON object with these fields: "
        f"{fields}. Do not add extra keys."
    )


def _vector_store_exists(persist_dir: Path) -> bool:
    if not persist_dir.exists():
        return False
    try:
        return any(persist_dir.iterdir())
    except OSError:
        return False


def _print_event(node_name: str, payload: object) -> None:
    header = node_name.replace("_", " ").upper()
    print(f"\n--- {header} ---")
    if not isinstance(payload, dict):
        print(payload)
        return
    if node_name == "retrieve":
        docs = payload.get("docs", [])
        print(f"Retrieved documents: {len(docs)}")
    elif node_name == "grade_documents":
        docs = payload.get("docs", [])
        scores = payload.get("scores", [])
        print(f"Relevant documents: {len(docs)} | Scores: {scores}")
    elif node_name == "transform_query":
        print(f"Rewritten query: {payload.get('query')}")
    elif node_name == "generate":
        print(payload.get("answer"))
    elif node_name == "check_hallucination":
        print(f"Hallucination grade: {payload.get('hallucination_grade')}")
    else:
        print(payload)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the RAG workflow from the CLI."""
    args = _parse_args(argv)
    settings = Settings.from_env()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading documents for mode=%s", args.mode)
    persist_dir = Path(settings.chroma_persist_dir)
    vector_db_exists = _vector_store_exists(persist_dir)
    chunks: list[DocumentChunk] = []
    if args.file is not None:
        if vector_db_exists and not args.force:
            logger.info(
                "Vector DB already exists at %s; skipping ingestion. Use --force to re-ingest.",
                persist_dir,
            )
        else:
            chunks = _load_chunks(args.file)
            logger.info("Loaded %s chunks from %s", len(chunks), args.file)

    retriever = HybridRetriever(
        vector_store=VectorStore(persist_directory=str(persist_dir)),
        keyword_store=KeywordStore(),
    )
    if chunks:
        retriever.add_documents(chunks)
        logger.info("Retriever initialized with %s chunks", len(chunks))
    else:
        logger.warning("No chunks loaded; retrieval may return empty results.")

    llm: BaseChatModel
    if args.mode == "cloud":
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = CloudChatModel(model_name=model_name)
    elif args.mode == "deepseek":
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        llm = DeepseekChatModel(model_name=model_name)
    else:
        llm = LocalHeuristicChatModel()
    generate_chain = _build_generate_chain(llm)
    app = build_workflow(llm, retriever=retriever, generate_chain=generate_chain)

    query = args.query or _prompt_for_query()
    logger.info("Running workflow for query")
    final_answer: str | None = None
    for event in app.stream({"query": query}):
        for node_name, payload in event.items():
            _print_event(node_name, payload)
            if isinstance(payload, dict) and "answer" in payload:
                final_answer = payload.get("answer")

    print("\n=== Answer ===")
    print(final_answer or "No answer generated.")


if __name__ == "__main__":
    main()
