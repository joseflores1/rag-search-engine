import math
import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    STOPWORDS_PATH,
    Movie,
    SearchResult,
    format_search_result,
    load_movies,
)


# common
def preprocess_text(text: str) -> str:
    text = text.lower()

    text = text.translate(str.maketrans("", "", string.punctuation))

    return text


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return [preprocess_text(word) for word in f.read().splitlines()]


STOPWORDS: list[str] = load_stopwords()


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()

    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)

    filtered_words = []
    for word in valid_tokens:
        if word not in STOPWORDS:
            filtered_words.append(word)

    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return stemmed_words


def tokenize_term(term: str) -> str:
    tokenized_term = tokenize_text(term)
    if len(tokenized_term) != 1:
        raise ValueError("term must be a single token")
    return tokenized_term[0]


class InvertedIndex:
    def __init__(self) -> None:
        self.index: defaultdict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies: defaultdict[int, Counter[str]] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        documents_ids = self.index[term]
        return sorted(documents_ids)

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        if not os.path.isdir(CACHE_DIR):
            os.mkdir(CACHE_DIR)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]

    def get_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[term])
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        n_docs = len(self.docmap)
        df = len(self.index[term])
        bm25_idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        doc_len = self.doc_lengths.get(doc_id, 0)
        avg_len = self.__get_avg_doc_length()

        if avg_len > 0:
            len_norm = 1 - b + b * (doc_len / avg_len)
        else:
            len_norm = 1

        bm25_tf = tf * (k1 + 1) / (tf + k1 * len_norm)
        return bm25_tf

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SearchResult]:
        tokens = tokenize_text(query)
        scores: dict[int, float] = {}

        for doc_id in self.docmap:
            score = 0.0
            for token in tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results: list[SearchResult] = []

        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formated_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formated_result)
        return results


# search command
def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    index = InvertedIndex()
    index.load()

    seen_docs: set[int] = set()
    seen_tokens: set[str] = set()
    results: list[Movie] = []

    query_tokens = tokenize_text(query)

    for token in query_tokens:
        if token in seen_tokens:
            continue
        seen_tokens.add(token)

        docs_ids = index.get_documents(token)

        for id in docs_ids:
            if id in seen_docs:
                continue
            seen_docs.add(id)

            results.append(index.docmap[id])
            if len(results) == limit:
                return results

    return results


# bm25 search command
def bm25_search_command(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[SearchResult]:
    idx = InvertedIndex()
    idx.load()
    docs = idx.bm25_search(query, limit)
    return docs


# build command
def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


# term frequency command
def tf_command(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    index.load()
    return index.get_tf(doc_id, tokenize_term(term))


# inverse document frequency command
def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_idf(tokenize_term(term))
    return idf


# TF-IDF command
def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, tokenize_term(term))


# BM"% IDF command
def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(tokenize_term(term))


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, tokenize_term(term), k1, b)
