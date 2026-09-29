import os
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    HF_TOKEN,
    Movie,
    SearchResult,
    format_search_result,
    load_movies,
)

EmbeddingArray = NDArray[Any]


class MultiModalSearch:
    def __init__(
        self, documents: list[Movie] | None = None, model_name: str = "clip-ViT-B-32"
    ) -> None:
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        self.documents = [] if documents is None else documents
        self.model = SentenceTransformer(model_name, token=HF_TOKEN)
        self.texts: list[str] = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts)
        self.norms: EmbeddingArray | None = None

    def embed_image(self, img_path: str) -> EmbeddingArray:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image = Image.open(img_path)
        image_embedding = self.model.encode([image])
        return image_embedding[0]

    def search_with_image(
        self, img_path: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SearchResult]:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image_embedding = self.embed_image(img_path)

        self.norms = np.linalg.norm(self.text_embeddings, axis=1)
        img_norm = np.linalg.norm(image_embedding)

        if img_norm == 0:
            scores_list: list[float] = [0.0 for _ in range(len(self.text_embeddings))]
        else:
            dot_products = self.text_embeddings @ image_embedding
            denominators = self.norms * img_norm
            scores = np.divide(
                dot_products,
                denominators,
                out=np.zeros_like(dot_products),
                where=denominators != 0,
            )
            scores_list = scores.tolist()

        idx_scores: list[tuple[int, float]] = [(i, score) for i, score in enumerate(scores_list)]
        idx_scores.sort(key=lambda x: x[1], reverse=True)
        results = [
            format_search_result(
                doc_id=self.documents[idx]["id"],
                title=self.documents[idx]["title"],
                document=self.documents[idx]["description"],
                score=score,
            )
            for idx, score in idx_scores[:limit]
        ]

        return results


def verify_image_embedding(img_path: str) -> None:
    search = MultiModalSearch()
    embedding = search.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(img_path: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchResult]:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found {img_path}")
    searcher = MultiModalSearch(documents=load_movies())
    return searcher.search_with_image(img_path, limit=limit)
