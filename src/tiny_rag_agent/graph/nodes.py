"""LangGraph node functions (state -> chain -> state)."""

from __future__ import annotations

from typing import Iterable, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from tiny_rag_agent.graph.chains import Grade, grader_chain, rewrite_chain
from tiny_rag_agent.graph.state import GraphState
from tiny_rag_agent.ingestion.schema import DocumentChunk
from tiny_rag_agent.retrieval.hybrid import HybridRetriever


def _format_documents(chunks: Iterable[DocumentChunk]) -> str:
    """Flatten document chunks into a single prompt-friendly string."""
    return "\n\n".join(chunk.text for chunk in chunks)


class HallucinationGrade(BaseModel):
    """Grounding check output with explanation."""

    score: Literal["yes", "no"] = Field(
        ...,
        description="Binary grounding verdict: 'yes' if grounded.",
    )
    explanation: str = Field(
        ...,
        description="Brief explanation with evidence or missing facts.",
    )


def retrieve(state: GraphState, retriever: HybridRetriever | None = None) -> dict[str, object]:
    """Retrieve documents for the current query.

    Args:
        state: Current graph state.
        retriever: Hybrid retriever instance (optional injection).

    Returns:
        Dict of state updates with retrieved documents.
    """
    query = state.query
    if not query:
        return {"docs": [], "scores": [], "answer": None}

    retriever = retriever or HybridRetriever()
    docs = retriever.search(query)
    return {"docs": docs, "scores": [], "answer": None}


def grade_documents(
    state: GraphState,
    llm: BaseChatModel | None = None,
    chain: Runnable | None = None,
) -> dict[str, object]:
    """Score relevance for each retrieved chunk and filter irrelevant docs.

    Args:
        state: Current graph state.
        llm: Chat model injected from configuration.
        chain: Optional prebuilt grader chain.

    Returns:
        Dict of state updates with filtered documents and scores.
    """
    docs = list(state.docs)
    if not docs:
        return {"docs": [], "scores": []}

    if chain is None and llm is not None:
        chain = grader_chain(llm)

    if chain is None:
        return {"docs": docs, "scores": [1.0] * len(docs)}

    filtered_docs: list[DocumentChunk] = []
    scores: list[float] = []
    for doc in docs:
        result = chain.invoke({"question": state.query, "document": doc.text})
        verdict = result.score if isinstance(result, Grade) else str(result)
        if verdict == "yes":
            filtered_docs.append(doc)
            scores.append(1.0)

    return {"docs": filtered_docs, "scores": scores}


def transform_query(
    state: GraphState,
    llm: BaseChatModel | None = None,
    chain: Runnable | None = None,
) -> dict[str, object]:
    """Rewrite the query to improve retrieval.

    Args:
        state: Current graph state.
        llm: Chat model injected from configuration.
        chain: Optional prebuilt rewrite chain.

    Returns:
        Dict of state updates with rewritten query.
    """
    if not state.query:
        return {"query": state.query}

    if chain is None and llm is not None:
        chain = rewrite_chain(llm)

    if chain is None:
        return {"query": state.query}

    rewritten = chain.invoke({"question": state.query})
    if not isinstance(rewritten, str):
        rewritten = str(rewritten)
    return {"query": rewritten.strip()}


def generate(
    state: GraphState,
    generate_chain: Runnable | None = None,
) -> dict[str, object]:
    """Generate an answer grounded in retrieved docs.

    Args:
        state: Current graph state.
        generate_chain: Prebuilt generation chain.

    Returns:
        Dict of state updates with generated answer.
    """
    if not state.docs:
        return {"answer": ""}

    if generate_chain is None:
        return {"answer": ""}

    documents = _format_documents(state.docs)
    output = generate_chain.invoke(
        {
            "question": state.query,
            "documents": documents,
        }
    )
    return {"answer": str(output)}


def check_hallucination(
    state: GraphState,
    llm: BaseChatModel | None = None,
    chain: Runnable | None = None,
) -> dict[str, object]:
    """Check if the generated answer is grounded in the documents.

    Args:
        state: Current graph state.
        llm: Chat model injected from configuration.
        chain: Optional prebuilt hallucination chain.

    Returns:
        Dict of state updates with hallucination_grade.
    """
    if not state.docs or not state.answer:
        return {"hallucination_grade": "hallucinated"}

    if chain is None and llm is not None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a strict grounding checker. "
                    "Answer 'yes' only if every factual claim in the answer "
                    "is directly supported by the documents. "
                    "If any claim is unsupported or missing, answer 'no'. "
                    "Use exact quotes from the documents as evidence in the explanation.",
                ),
                (
                    "human",
                    "Documents:\n{documents}\n\nAnswer:\n{generation}\n\n"
                    "Does the answer contain any information NOT present in the documents?",
                ),
            ]
        )
        chain = prompt | llm.with_structured_output(HallucinationGrade)

    if chain is None:
        return {"hallucination_grade": "hallucinated"}

    documents = _format_documents(state.docs)
    result = chain.invoke({"documents": documents, "generation": state.answer})
    score = result.score if isinstance(result, HallucinationGrade) else str(result)
    grade = "grounded" if score == "yes" else "hallucinated"
    return {"hallucination_grade": grade}
