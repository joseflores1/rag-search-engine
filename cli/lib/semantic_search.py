import json
import os
import re
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .search_utils import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    HF_TOKEN,
    MOVIE_EMBEDDINGS_PATH,
    Movie,
    load_movies,
)

EmbeddingArray = NDArray[Any]


class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-V2") -> None:
        self.model = SentenceTransformer(model_name, token=HF_TOKEN)
        self.embeddings: EmbeddingArray | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str) -> EmbeddingArray:
        if len(text) == 0 or len(text.strip()) == 0:
            raise ValueError("text must not be empty or contain only whitespace")
        embedding = self.model.encode([text])[0]
        return embedding

    def build_embeddings(self, documents: list[Movie]) -> EmbeddingArray:

        self.documents = documents
        self.document_map = {}
        doc_descriptions: list[str] = []

        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_descriptions.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(doc_descriptions, show_progress_bar=True)

        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> EmbeddingArray:
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if not os.path.exists(MOVIE_EMBEDDINGS_PATH):
            return self.build_embeddings(documents)

        self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)

        if len(self.embeddings) != len(documents):
            return self.build_embeddings(documents)

        return self.embeddings

    def search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SemanticSearchResult]:
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if len(self.document_map) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        embed_query = self.generate_embedding(query)
        scores_docs: list[tuple[float, Movie]] = []

        for i, embedding in enumerate(self.embeddings, 1):
            score = cosine_similarity(embed_query, embedding)
            scores_docs.append((score, self.document_map[i]))

        scores_docs.sort(key=lambda x: x[0], reverse=True)

        results: list[SemanticSearchResult] = []

        for score, doc in scores_docs[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: EmbeddingArray | None = None
        self.chunk_metadata: list[ChunkMetadata] | None = None

    def build_chunk_embeddings(self, documents: list[Movie]) -> EmbeddingArray:

        self.documents = documents
        self.document_map = {}

        chunks: list[str] = []
        chunks_metadata: list[ChunkMetadata] = []

        for doc in self.documents:
            self.document_map[doc["id"]] = doc

            if len(doc["description"].strip()) == 0:
                continue

            doc_chunks = fixed_size_chunking(
                doc["description"],
                True,
                DEFAULT_SEMANTIC_CHUNK_SIZE,
                DEFAULT_CHUNK_OVERLAP,
            )

            for i in range(len(doc_chunks)):
                chunks.append(doc_chunks[i])
                chunks_metadata.append(
                    {
                        "movie_idx": doc["id"],
                        "chunk_idx": i,
                        "total_chunks": len(doc_chunks),
                    }
                )

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)

        os.makedirs(os.path.dirname(CHUNK_METADATA_PATH), exist_ok=True)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {
                    "chunks": chunks_metadata,
                    "total_chunks": len(chunks),
                },
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> EmbeddingArray:
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if not os.path.exists(CHUNK_EMBEDDINGS_PATH) or not os.path.exists(
            CHUNK_METADATA_PATH
        ):
            return self.build_chunk_embeddings(documents)

        self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
        with open(CHUNK_METADATA_PATH, "r") as f:
            self.chunk_metadata = json.load(f)["chunks"]

        if len(self.chunk_embeddings) != len(self.chunk_metadata):
            raise RuntimeError("Something went wrong")

        return self.chunk_embeddings


# Commands
def verify_model() -> None:
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")


def embed_text(text: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    search = SemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    scores_docs = search_instance.search(query, limit)
    for i, res in enumerate(scores_docs, 1):
        print(f"{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"  {res['description'][:100]}\n")


def fixed_size_chunking(
    text: str, semantic: bool, chunk_size: int, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must be equal or greater than 0")

    if semantic:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = [
            " ".join(sentences[i : i + chunk_size])
            for i in range(0, len(sentences), chunk_size - overlap)
            if len(sentences[i : i + chunk_size]) > overlap
        ]
    else:
        words = text.split()
        chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size - overlap)
        ]

    return chunks


def chunk_text(
    text: str, semantic: bool, chunk_size: int, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> None:
    chunks = fixed_size_chunking(text, semantic, chunk_size, overlap)
    print(
        f"{'Semantically c' if semantic else 'C'}hunking {len(text.strip())} characters"
    )
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def embed_chunks() -> EmbeddingArray:
    searcher = ChunkedSemanticSearch()
    movies = load_movies()
    chunk_embeddings = searcher.load_or_create_chunk_embeddings(movies)
    return chunk_embeddings


# Score metric
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
