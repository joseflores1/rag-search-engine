import os
import pickle
import string
import sys
from collections import Counter, defaultdict

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

def tokenize_term(term: str) -> str:
    tokenized_term = tokenize_text(term)
    if len(tokenized_term) != 1:
        raise RuntimeError("tokenization of single term did not produce only one token")
    return tokenized_term

# search command
def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    index = InvertedIndex()
    index.load()

    seen_docs, seen_tokens, results = set(), set(), []
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


# build command
class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter) 
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id] += {token:1}


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

        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)


    def load(self) -> None:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]

# build command
def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

# term frequency command
def tf_command(doc_id: int, term: str) -> None:
    index = InvertedIndex()
    index.load()
    tokenized_term = tokenize_term(term)
    term_freq = index.term_frequencies[doc_id][tokenized_term[0]]
    print(f"'{term}' appears {term_freq} times in document {doc_id}")