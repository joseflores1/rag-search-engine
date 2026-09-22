import os
from typing import Literal, TypedDict

from .keyword_search import InvertedIndex
from .llm_utils import enhance_query, rerank_docs, rerank_limit
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    Movie,
    RRFSearchResult,
    SearchResult,
    load_movies,
)
from .semantic_search import ChunkedSemanticSearch


class HybridSearchResult(TypedDict):
    id: int
    title: str
    description: str
    key_score: float
    semantic_score: float
    hybrid_score: float


class RRFResults(TypedDict):
    results: list[RRFSearchResult]
    enhanced_query: str | None


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        if not self.idx.index:
            self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self,
        query: str,
        alpha: float = DEFAULT_ALPHA,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[HybridSearchResult]:
        bm25_results = self._bm25_search(query, 500 * limit)
        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)

        hybrid_results = combine_search_results(bm25_results, semantic_results, alpha)

        return hybrid_results[:limit]

    def rrf_search(
        self, query: str, k: int = RRF_K, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[RRFSearchResult]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        rrf_results = reciprocal_rank_fusion(bm25_results, semantic_results, k)
        return rrf_results[:limit]


# Helper
def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1 / (k + rank)


def normalize_scores(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return []

    max_value = max(scores)
    min_value = min(scores)

    if max_value != min_value:
        new_scores = [(score - min_value) / (max_value - min_value) for score in scores]
        return new_scores

    return [1.0 for _ in range(len(scores))]


def combine_search_results(
    bm25_results: list[SearchResult],
    semantic_results: list[SearchResult],
    alpha: float = DEFAULT_ALPHA,
) -> list[HybridSearchResult]:

    keyword_norms = normalize_scores([res["score"] for res in bm25_results])
    semantic_norms = normalize_scores([res["score"] for res in semantic_results])

    hybrid_results: list[HybridSearchResult] = []

    search_lookup = {item["id"]: (keyword_norms[i], item) for i, item in enumerate(bm25_results)}

    for i, semantic in enumerate(semantic_results):
        matched_search = search_lookup.get(semantic["id"])

        if matched_search:
            key_score, match = matched_search
            semantic_score = semantic_norms[i]
            del search_lookup[semantic["id"]]
            hybrid_results.append(
                {
                    "id": semantic["id"],
                    "title": semantic["title"],
                    "description": semantic["document"],
                    "key_score": key_score,
                    "semantic_score": semantic_score,
                    "hybrid_score": hybrid_score(key_score, semantic_score, alpha),
                }
            )
        else:
            hybrid_results.append(
                {
                    "id": semantic["id"],
                    "title": semantic["title"],
                    "description": semantic["document"],
                    "key_score": 0.0,
                    "semantic_score": semantic_norms[i],
                    "hybrid_score": hybrid_score(0.0, semantic_norms[i], alpha),
                }
            )

    for key in search_lookup:
        key_score = search_lookup[key][0]
        match = search_lookup[key][1]
        hybrid_results.append(
            {
                "id": match["id"],
                "title": match["title"],
                "description": match["document"],
                "key_score": key_score,
                "semantic_score": 0.0,
                "hybrid_score": hybrid_score(key_score, 0.0, alpha),
            }
        )

    hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return hybrid_results


def reciprocal_rank_fusion(
    bm25_results: list[SearchResult], semantic_results: list[SearchResult], k: int = RRF_K
) -> list[RRFSearchResult]:
    rrf_results: list[RRFSearchResult] = []
    search_lookup = {item["id"]: (i, item) for i, item in enumerate(bm25_results, 1)}

    for i, result in enumerate(semantic_results, 1):
        matched_search = search_lookup.get(result["id"])
        if matched_search:
            del search_lookup[result["id"]]
            bm25_rank = matched_search[0]
            match = matched_search[1]
            rrf_results.append(
                {
                    "id": match["id"],
                    "title": match["title"],
                    "description": match["document"],
                    "rrf_score": rrf_score(i, k) + rrf_score(bm25_rank, k),
                    "bm25_rank": bm25_rank,
                    "chunk_rank": i,
                    "rerank_score": 0.0,
                }
            )
        else:
            rrf_results.append(
                {
                    "id": result["id"],
                    "title": result["title"],
                    "description": result["document"],
                    "rrf_score": rrf_score(i, k),
                    "bm25_rank": 0,
                    "chunk_rank": i,
                    "rerank_score": 0.0,
                }
            )

    for id in search_lookup:
        bm25_rank = search_lookup[id][0]
        result = search_lookup[id][1]
        rrf_results.append(
            {
                "id": result["id"],
                "title": result["title"],
                "description": result["document"],
                "rrf_score": rrf_score(bm25_rank, k),
                "bm25_rank": bm25_rank,
                "chunk_rank": 0,
                "rerank_score": 0.0,
            }
        )

    rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return rrf_results


# Commands
def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[HybridSearchResult]:
    docs = load_movies()
    searcher = HybridSearch(docs)
    results = searcher.weighted_search(query, alpha, limit)
    return results


def rrf_search_command(
    query: str,
    k: int = RRF_K,
    limit: int = DEFAULT_SEARCH_LIMIT,
    enhance: Literal["spell", "rewrite", "expand"] | None = None,
    rerank_method: Literal["individual"] | None = None,
) -> RRFResults:
    docs = load_movies()
    searcher = HybridSearch(docs)

    if enhance:
        query = enhance_query(query, method=enhance)

    new_limit = rerank_limit(limit, rerank_method)
    results = searcher.rrf_search(query, k, new_limit)
    if rerank_method:
        results = rerank_docs(query, results, rerank_method)[:limit]

    return {"enhanced_query": query, "results": results}
