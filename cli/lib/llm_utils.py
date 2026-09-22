import os
import re
from time import sleep
from typing import Optional

from dotenv import get_key
from openai import OpenAI

from .search_utils import DEFAULT_SEARCH_LIMIT, ENV_PATH, RRFSearchResult

api_key = get_key(ENV_PATH, "OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
os.environ["OPENROUTER_API_KEY"] = api_key
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"


def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    corrected = (response.choices[0].message.content or "").strip().strip('"')
    return corrected if corrected else query


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep the rewritten query concise (under 10 words)
    - It should be a Google-style search query, specific enough to yield relevant results
    - Don't use boolean logic

    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    If you cannot improve the query, output the original unchanged.
    Output only the rewritten query text, nothing else.

    User query: "{query}"
    """  # noqa: E501

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    rewritten = (response.choices[0].message.content or "").strip().strip('"')
    return rewritten if rewritten else query


def expand_query(query: str) -> str:
    prompt = f"""Expand the user-provided movie search query below with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    Output only the additional terms; they will be appended to the original query.

    Examples:
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    User query: "{query}"
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    expanded = (response.choices[0].message.content or "").strip().strip('"')
    return f"{query} {expanded}".strip()


def enhance_query(query: str, method: Optional[str] = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand_query(query)
        case _:
            return query


# Rerank
_RERANK_SCORE_PATTERN = re.compile(r"^(?:[0-9](?:\.[0-9]+)?|10(?:\.0+)?)$")


def rerank_limit(
    limit: int = DEFAULT_SEARCH_LIMIT, rerank_method: Optional[str] | None = None
) -> int:
    match rerank_method:
        case "individual":
            return 5 * limit
        case _:
            return limit


def rerank_individual(query: str, doc: RRFSearchResult, max_attempts: int = 3) -> float:
    prompt = f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {doc.get("title", "")} - {doc.get("description", "")}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Output ONLY the number in your response, no other text or explanation.

    Score:"""

    for _ in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            score = (response.choices[0].message.content or "").strip().strip('"')
        except Exception:
            sleep(3)
            continue

        if _RERANK_SCORE_PATTERN.match(score):
            return float(score)

        sleep(3)

    return 0.0

def rerank_docs(
    query: str, results: list[RRFSearchResult], method: Optional[str] | None
) -> list[RRFSearchResult]:
    match method:
        case "individual":
            reranker = rerank_individual
        case _:
            return results

    for doc in results:
        rerank_score = reranker(query, doc)
        doc["rerank_score"] = rerank_score
        sleep(3)
    
    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results
