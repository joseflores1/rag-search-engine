from typing import TypedDict

from .client import get_client
from .hybrid_search import HybridSearch
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    SEARCH_MULTIPLIER,
    SearchResult,
    load_movies,
)


# --- Types definition ---
class RagErrorResult(TypedDict):
    query: str
    search_results: list[SearchResult]
    error: str


class RagResult(TypedDict):
    query: str
    search_results: list[SearchResult]
    answer: str


class SummaryResult(TypedDict):
    query: str
    summary: str
    search_results: list[SearchResult]


class SummaryErrorResult(TypedDict):
    query: str
    error: str


class CitationResult(TypedDict):
    query: str
    answer: str
    search_results: list[SearchResult]


class CitationErrorResult(TypedDict):
    query: str
    error: str


class QuestionResult(TypedDict):
    question: str
    answer: str
    search_results: list[SearchResult]


class QuestionErrorResult(TypedDict):
    question: str
    error: str
# --- RAG ---
def generate_answer(
    search_results: list[SearchResult], query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> str:
    client, model = get_client()

    context = "".join(
        [f"{result['title']}: {result['document']}\n\n" for result in search_results[:limit]]
    )

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {context}

    Answer:"""  # noqa: E501

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return (response.choices[0].message.content or "").strip()


def rag(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> RagErrorResult | RagResult:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER)

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    answer = generate_answer(search_results, query, limit)

    return {
        "query": query,
        "search_results": search_results[:limit],
        "answer": answer,
    }


def rag_command(query):
    return rag(query)

# --- Summary ---
def generate_summary(
    search_results: list[SearchResult], query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> str:
    client, model = get_client()

    results = "".join(
        [
            f"Document {i}: {result['title']}; {result['document']}\n\n"
            for i, result in enumerate(search_results[:limit], 1)
        ]
    )

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Webflyx users. Webflyx is a movie streaming service.

    Query: {query}

    Search results:
    {results}

Provide a comprehensive 3 to 4 sentence answer that combines information from multiple sources:"""  # noqa: E501

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return (response.choices[0].message.content or "").strip()


def summarize(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> SummaryErrorResult | SummaryResult:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER)

    if not search_results:
        return {
            "query": query,
            "error": "No results found",
        }

    summary = generate_summary(search_results, query, limit)

    return {
        "query": query,
        "search_results": search_results[:limit],
        "summary": summary,
    }


def summarize_command(query, limit: int = DEFAULT_SEARCH_LIMIT):
    return summarize(query, limit)

# -- Citations ---
def generate_answer_with_citations(
    search_results: list[SearchResult], query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> str:
    client, model = get_client()

    documents = "".join(
        [
            f"[{i}]: {result['title']}; {result['document']}\n\n"
            for i, result in enumerate(search_results[:limit], 1)
        ]
    )

    prompt = f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Webflyx, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {documents}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""  # noqa: E501
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return (response.choices[0].message.content or "").strip()


def citations(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> CitationErrorResult | CitationResult:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER)

    if not search_results:
        return {
            "query": query,
            "error": "No results found",
        }

    answer_with_citations = generate_answer_with_citations(search_results, query, limit)

    return {
        "query": query,
        "search_results": search_results[:limit],
        "answer": answer_with_citations,
    }


def citations_command(query, limit: int = DEFAULT_SEARCH_LIMIT):
    return citations(query, limit)

# --- Question ---
def generate_answer_of_question(
    search_results: list[SearchResult], query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> str:
    client, model = get_client()

    context = "".join(
        [
            f"[{i}]: {result['title']}; {result['document']}\n\n"
            for i, result in enumerate(search_results[:limit], 1)
        ]
    )

    prompt = f"""Answer the following question based on the provided documents.

    Question: {query}

    Documents:
    {context}

    General instructions:
    - Answer directly and concisely
    - Use only information from the documents
    - If the answer isn't in the documents, say "I don't have enough information"
    - Cite sources when possible
    - Be casual and conversational
    - Talk like a normal person would in a chat conversation

    Guidance on types of questions:
    - Factual questions: Provide a direct answer
    - Analytical questions: Compare and contrast information from the documents
    - Opinion-based questions: Acknowledge subjectivity and provide a balanced view

    Answer:"""

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return (response.choices[0].message.content or "").strip()


def question(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> QuestionErrorResult | QuestionResult:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER)

    if not search_results:
        return {
            "question": query,
            "error": "No results found",
        }

    answer_of_question = generate_answer_of_question(search_results, query, limit)

    return {
        "question": query,
        "search_results": search_results[:limit],
        "answer": answer_of_question,
    }


def question_command(query, limit: int = DEFAULT_SEARCH_LIMIT):
    return question(query, limit)