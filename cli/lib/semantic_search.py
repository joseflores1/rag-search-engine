import os
from typing import Any

import numpy as np
from lib.search_utils import CACHE_DIR, Movie, load_movies
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

EmbeddingArray = NDArray[Any]


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


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-V2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text: str) -> EmbeddingArray:
        if len(text) == 0 or len(text.strip()) == 0:
            raise ValueError("text must not be empty or contain only whitespace")
        embedding = self.model.encode([text], show_progress_bar=True)[0]
        return embedding

    def __fill_docs(self, documents: list[Movie]) -> list[str]:
        self.documents = documents
        doc_descriptions = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_descriptions.append(f"{doc['title']}: {doc['description']}")
        return doc_descriptions

    def build_embeddings(self, documents: list[Movie]) -> EmbeddingArray[EmbeddingArray]:
        doc_descriptions = self.__fill_docs(documents)
        
        self.embeddings = np.array([self.generate_embedding(description) for description in doc_descriptions])
        
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> list[EmbeddingArray]:
        self.__fill_docs(documents)

        if not os.path.exists(self.embeddings_path):
            return self.build_embeddings(documents)

        self.embeddings = np.load(self.embeddings_path)

        if len(self.embeddings) != len(documents):
            return self.build_embeddings(documents)

        return self.embeddings
