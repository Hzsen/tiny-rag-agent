"""LLM chain building blocks (LCEL)."""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field


class Grade(BaseModel):
    """Binary grading output for LLM evaluators."""

    score: Literal["yes", "no"] = Field(
        ...,
        description="Binary relevance verdict: 'yes' or 'no'.",
    )


class HallucinationGrade(BaseModel):
    """Binary grounding evaluation output."""

    score: Literal["yes", "no"] = Field(
        ...,
        description="Binary grounding verdict: 'yes' if grounded.",
    )


class AnswerGrade(BaseModel):
    """Binary answer-quality evaluation output."""

    score: Literal["yes", "no"] = Field(
        ...,
        description="Binary answer verdict: 'yes' if it addresses the question.",
    )


def grader_chain(llm: BaseChatModel) -> Runnable:
    """Build a relevance grader chain.

    Args:
        llm: Chat model injected from configuration.

    Returns:
        LCEL chain that yields a Grade model.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict relevance grader. Return only 'yes' or 'no'.",
            ),
            (
                "human",
                "Question:\n{question}\n\nDocument:\n{document}\n\n"
                "Is the document relevant to the question?",
            ),
        ]
    )
    return prompt | llm.with_structured_output(Grade)


def rewrite_chain(llm: BaseChatModel) -> Runnable:
    """Build a query rewrite chain optimized for vector retrieval.

    Args:
        llm: Chat model injected from configuration.

    Returns:
        LCEL chain that yields a rewritten query string.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the question to be concise and retrieval-optimized.",
            ),
            (
                "human",
                "Original question:\n{question}\n\nRewrite it for retrieval.",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def hallucination_grader_chain(llm: BaseChatModel) -> Runnable:
    """Build a grounding checker chain.

    Args:
        llm: Chat model injected from configuration.

    Returns:
        LCEL chain that yields a HallucinationGrade model.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Assess if the answer is fully grounded in the documents. "
                "Return only 'yes' or 'no'.",
            ),
            (
                "human",
                "Documents:\n{documents}\n\nAnswer:\n{generation}\n\n"
                "Is the answer grounded in the documents?",
            ),
        ]
    )
    return prompt | llm.with_structured_output(HallucinationGrade)


def answer_grader_chain(llm: BaseChatModel) -> Runnable:
    """Build an answer-quality checker chain.

    Args:
        llm: Chat model injected from configuration.

    Returns:
        LCEL chain that yields an AnswerGrade model.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Assess if the answer addresses the question. "
                "Return only 'yes' or 'no'.",
            ),
            (
                "human",
                "Question:\n{question}\n\nAnswer:\n{generation}\n\n"
                "Does the answer address the question?",
            ),
        ]
    )
    return prompt | llm.with_structured_output(AnswerGrade)
