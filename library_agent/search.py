"""Intelligent book search.

A normal library search only finds exact words. This one ranks books by how
*similar* they are to the query, using TF-IDF vectors and cosine similarity.

How it works
------------
1. Every book becomes one text "document": title (repeated, so it counts more),
   author, genre and description. Documents are cleaned and stemmed.
2. TF-IDF turns each document into a vector. Words that are rare across the
   catalogue (e.g. "pathology") get high weight, common words get low weight.
3. The query is cleaned the same way, abbreviations are expanded ("ml" ->
   "machine learning"), and turned into a vector.
4. Cosine similarity measures the angle between the query vector and each book
   vector. Small angle = similar = high score.
5. Typing mistakes are handled twice. First, each query word that is not in the
   catalogue vocabulary is replaced by its closest vocabulary word
   ("lerning" -> "learning"). Second, a character-level TF-IDF (3-4 letter
   chunks) over titles and author names catches misspelt names.
   Final score = 0.7 x word score + 0.3 x character score.
"""
import difflib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import nlp
from .config import SEARCH_MIN_SCORE

WORD_WEIGHT = 0.7
CHAR_WEIGHT = 0.3


class BookSearch:
    def __init__(self, books: pd.DataFrame):
        self.books = books.reset_index(drop=True)

        # Word-level index (with word pairs, so "machine learning" is one feature).
        word_docs = [
            nlp.normalize(f"{r.title} {r.title} {r.title} {r.author} {r.genre} {r.description}")
            for r in self.books.itertuples()
        ]
        self._doc_terms = [set(d.split()) for d in word_docs]
        # Every word the catalogue contains: the dictionary for spelling correction.
        self.vocab = sorted({
            w for r in self.books.itertuples()
            for w in nlp.words(f"{r.title} {r.author} {r.genre} {r.description}")
        })
        self._vocab_set = set(self.vocab)
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.word_matrix = self.word_vec.fit_transform(word_docs)

        # Character-level index over the short fields only (title, author, genre).
        char_docs = [nlp.clean(f"{r.title} {r.author} {r.genre}") for r in self.books.itertuples()]
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True)
        self.char_matrix = self.char_vec.fit_transform(char_docs)

    def correct_query(self, query: str) -> str:
        """Fix likely typos: unknown words become their closest catalogue word."""
        fixed = []
        for w in nlp.raw_words(query):
            known = w in self._vocab_set or w in nlp.STOP_WORDS or w in nlp.SYNONYMS
            if known or len(w) < 4:
                fixed.append(w)
                continue
            close = difflib.get_close_matches(w, self.vocab, n=1, cutoff=0.8)
            fixed.append(close[0] if close else w)
        return " ".join(fixed)

    def scores(self, query: str):
        """Similarity of the query to every book (numpy array, same order as self.books)."""
        word_q = self.word_vec.transform([nlp.normalize(nlp.expand_query(self.correct_query(query)))])
        char_q = self.char_vec.transform([nlp.clean(query)])
        return (WORD_WEIGHT * cosine_similarity(word_q, self.word_matrix)[0]
                + CHAR_WEIGHT * cosine_similarity(char_q, self.char_matrix)[0])

    def search(self, query: str, top_n: int = 10, genre: str | None = None,
               available_only: bool = False, availability: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return the best matching books, best first.

        ``availability`` (from db.availability_df) adds an ``available`` column and
        lets ``available_only`` hide books with no free copy.
        """
        if not query.strip():
            return pd.DataFrame(columns=["book_id", "title", "author", "genre", "score", "matched"])

        result = self.books[["book_id", "title", "author", "genre"]].copy()
        result["score"] = self.scores(query)

        q_terms = set(nlp.tokenize(nlp.expand_query(self.correct_query(query))))
        result["matched"] = [", ".join(sorted(q_terms & terms)) for terms in self._doc_terms]

        if availability is not None:
            result = result.merge(availability[["book_id", "available"]], on="book_id", how="left")
            if available_only:
                result = result[result["available"] > 0]
        if genre:
            result = result[result["genre"] == genre]

        result = result[result["score"] >= SEARCH_MIN_SCORE]
        result = result.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
        result.attrs["corrected_query"] = self.correct_query(query)  # so the UI can say "showing results for ..."
        return result