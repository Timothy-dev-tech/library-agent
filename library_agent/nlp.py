"""Shared text processing used by search, the recommender and the chatbot.

Pipeline:  raw text -> lower-case, remove accents -> split into words
           -> drop stop words -> stem ("recommendations" -> "recommend").

Stemming means "learn", "learning" and "learned" all count as the same word,
which makes search and similarity far more forgiving.

Note: we use our OWN short stop-word list. scikit-learn's built-in list
contains the words "computer" and "system", which would delete the most
important words in a library catalogue.
"""
import re
import unicodedata

from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()
_WORD = re.compile(r"[a-z0-9]+")

STOP_WORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it its
me my of on or our please so some than that the their them then there these they this to us was we were
what when where which who whom why will with would you your about any also just
""".split())

# Rule-based query expansion: students type abbreviations and slang.
SYNONYMS = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "dsa": "data structures algorithms",
    "os": "operating system",
    "db": "database",
    "dbms": "database",
    "stats": "statistics",
    "math": "mathematics",
    "maths": "mathematics",
    "econ": "economics",
    "bio": "biology",
    "chem": "chemistry",
    "coding": "programming",
    "code": "programming",
    "novel": "fiction",
    "novels": "fiction",
    "story": "fiction",
    "stories": "fiction",
    "study": "learning",
    "motivation": "habits",
    "colonial": "colonialism",
    "intro": "introduction",
    # Stemming does not join country and nationality words, so we link them by hand.
    "kenyan": "kenya",
    "kenya": "kenyan",
    "african": "africa",
    "africa": "african",
    "nigerian": "nigeria",
    "nigeria": "nigerian",
}


def strip_accents(text: str) -> str:
    """'Ngũgĩ' -> 'Ngugi' so that people can type names without special letters."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def clean(text: str) -> str:
    """Lower-case and accent-free, nothing else (used for character n-grams)."""
    return strip_accents(text).lower()


def raw_words(text: str) -> list[str]:
    """Every lower-case word, nothing removed (used for spelling correction)."""
    return _WORD.findall(clean(text))


def words(text: str) -> list[str]:
    """Lower-case words with stop words removed (no stemming)."""
    return [w for w in _WORD.findall(clean(text)) if w not in STOP_WORDS]


def tokenize(text: str) -> list[str]:
    """Words, stop words removed, stemmed."""
    return [_stemmer.stem(w) for w in words(text)]


def normalize(text: str) -> str:
    """Tokenised text joined back into one string, ready for TF-IDF."""
    return " ".join(tokenize(text))


def expand_query(text: str) -> str:
    """Add the long form of abbreviations: 'ml basics' -> 'ml basics machine learning'."""
    extra = [SYNONYMS[w] for w in _WORD.findall(clean(text)) if w in SYNONYMS]
    return " ".join([text, *extra])