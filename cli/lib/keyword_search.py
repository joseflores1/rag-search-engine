import os
import pickle
import string
from collections import defaultdict

from nltk.stem import PorterStemmer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, STOPWORDS_PATH, load_movies


def preprocess_text(text: str) -> str:
    text = text.lower()

    text = text.translate(str.maketrans("", "", string.punctuation))

    return text

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return [preprocess_text(word) for word in f.read().splitlines()]


STOPWORDS = load_stopwords()

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

# search command
def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []

    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])

        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


# build command
class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)

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


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")
