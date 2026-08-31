import os
from typing import TypedDict

from .keyword_search import InvertedIndex
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    DOCUMENT_PREVIEW_LENGTH,
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

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT) -> list[HybridSearchResult]:
        keyword_results = self._bm25_search(query, 500 * limit)
        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)

        hybrid_results: list[HybridSearchResult] = []

        search_lookup = {item["id"]: item for item in keyword_results}

        for semantic in semantic_results:
            matched_search = search_lookup.get(semantic["id"])

            if matched_search:
                hybrid_results.append({
                    "id": semantic["id"],
                    "title": semantic["title"],
                    "description": semantic["document"],
                    "key_score": matched_search["score"],
                    "semantic_score": semantic["score"]
                })

        key_scores = normalize_scores([res["key_score"] for res in hybrid_results])
        semantic_scores = normalize_scores([res["semantic_score"] for res in hybrid_results])

        for i in range(len(hybrid_results)):
            hybrid_results[i]["hybrid_score"] = hybrid_score(key_scores[i], semantic_scores[i], alpha)
            hybrid_results[i]["key_score"] = key_scores[i]
            hybrid_results[i]["semantic_score"] = semantic_scores[i]

        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_results[:limit]


    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

# Helper
def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

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
def hybrich_search(query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    docs = load_movies()
    searcher = HybridSearch(docs)
    results = searcher.weighted_search(query, alpha, limit)
    for i, res in enumerate(results, 1):

        print(f"{i}. {res["title"]}")
        print(f" Hybrid score: {res["hybrid_score"]:.3f}")
        print(f" BM25: {res["key_score"]:.3f}, Semantic: {res["semantic_score"]:.3f}")
        print(f" {res["description"][:DOCUMENT_PREVIEW_LENGTH]}")