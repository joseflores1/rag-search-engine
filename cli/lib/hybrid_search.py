import os
from typing import TypedDict

from .keyword_search import InvertedIndex
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    DOCUMENT_PREVIEW_LENGTH,
    RRF_K,
    Movie,
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


class RRFSearchResult(TypedDict):
    id: int
    title: str
    description: str
    rrf_score: float
    bm25_rank: int
    chunk_rank: int


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
        keyword_results = self._bm25_search(query, 500 * limit)
        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)

        keyword_norms = normalize_scores([res["score"] for res in keyword_results])
        semantic_norms = normalize_scores([res["score"] for res in semantic_results])

        hybrid_results: list[HybridSearchResult] = []

        search_lookup = {
            item["id"]: (keyword_norms[i], item) for i, item in enumerate(keyword_results)
        }

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

        for key_score, match in search_lookup.values():
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
        return hybrid_results[:limit]

    def rrf_search(
        self, query: str, k: int = RRF_K, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[RRFSearchResult]:
        bm25_results = self._bm25_search(query, limit * 500)
        chunk_results = self.semantic_search.search_chunks(query, limit * 500)
        rrf_results: list[RRFSearchResult] = []
        search_lookup = {item["id"]: (i, item) for i, item in enumerate(bm25_results, 1)}

        for i, result in enumerate(chunk_results, 1):
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
                    }
                )

        for key in search_lookup:
            bm25_rank = search_lookup[key][0]
            result = search_lookup[key][1]
            rrf_results.append(
                {
                    "id": result["id"],
                    "title": result["title"],
                    "description": result["document"],
                    "rrf_score": rrf_score(bm25_rank, k),
                    "bm25_rank": bm25_rank,
                    "chunk_rank": 0,
                }
            )

       
        rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
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


# Commands
def hybrich_search(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[HybridSearchResult]:
    docs = load_movies()
    searcher = HybridSearch(docs)
    results = searcher.weighted_search(query, alpha, limit)
    return results

def rrf_search_command(query: str, k: int = RRF_K, limit: int = DEFAULT_SEARCH_LIMIT) -> list[RRFSearchResult]:
    docs = load_movies()
    searcher = HybridSearch(docs)
    results = searcher.rrf_search(query, k, limit)
    return results
