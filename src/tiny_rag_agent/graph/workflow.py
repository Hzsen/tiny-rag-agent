"""LangGraph workflow structure (nodes + edges)."""

from __future__ import annotations

from typing import Iterable

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph

from tiny_rag_agent.graph.nodes import (
    check_hallucination,
    generate,
    grade_documents,
    retrieve,
    transform_query,
)
from tiny_rag_agent.graph.state import GraphState
from tiny_rag_agent.retrieval.hybrid import HybridRetriever


def build_workflow(
    llm: BaseChatModel,
    retriever: HybridRetriever | None = None,
    generate_chain: Runnable | None = None,
):
    """Create and compile the LangGraph workflow.

    Args:
        llm: Chat model injected from configuration.
        retriever: Optional hybrid retriever injection.
        generate_chain: Optional generation chain.

    Returns:
        Compiled LangGraph app.
    """
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", lambda state: retrieve(state, retriever=retriever))
    graph.add_node(
        "grade_documents",
        lambda state: grade_documents(state, llm=llm),
    )
    graph.add_node("transform_query", lambda state: transform_query(state, llm=llm))
    graph.add_node(
        "generate",
        lambda state: generate(state, generate_chain=generate_chain),
    )
    graph.add_node("check_hallucination", lambda state: check_hallucination(state, llm=llm))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")

    def _route_after_grade(state: GraphState) -> str:
        """Decide whether to rewrite or generate after grading."""
        # 看看 state.docs 里还有没有文档？
        # 如果空了（说明都被过滤掉了），去 "transform_query"（改写问题）。
        # 如果还有，去 "generate"（生成答案）。
        return "transform_query" if not state.docs else "generate"

    graph.add_conditional_edges(
        "grade_documents",
        _route_after_grade,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )

    graph.add_edge("transform_query", "retrieve")

    graph.add_edge("generate", "check_hallucination")

    def _route_after_check(state: GraphState) -> str:
        """Route based on hallucination grade."""
        return "end" if state.hallucination_grade == "grounded" else "generate"

    graph.add_conditional_edges(
        "check_hallucination",
        _route_after_check,
        {
            "generate": "generate",
            "end": END,
        },
    )

    return graph.compile()
