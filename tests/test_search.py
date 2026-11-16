import pytest

from library_agent import db
from library_agent.evaluation import evaluate_search
from library_agent.search import BookSearch

ML_BOOK = "Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow"


@pytest.fixture(scope="module")
def engine(books):
    return BookSearch(books)


def test_topic_query_finds_the_right_book(engine):
    assert engine.search("machine learning").iloc[0]["title"] == ML_BOOK


def test_typos_are_corrected(engine):
    assert engine.correct_query("machin lerning") == "machine learning"
    assert engine.search("machin lerning").iloc[0]["title"] == ML_BOOK


def test_abbreviations_are_expanded(engine):
    assert engine.search("AI").iloc[0]["title"] == "Artificial Intelligence: A Modern Approach"


def test_accents_do_not_matter(engine):
    titles = set(engine.search("Ngũgĩ wa Thiong'o", top_n=5)["title"])
    assert {"A Grain of Wheat", "The River Between"} <= titles


def test_nonsense_query_returns_nothing(engine):
    assert engine.search("xyzzy qwerty").empty


def test_blank_query_returns_nothing(engine):
    assert engine.search("   ").empty


def test_genre_filter(engine):
    result = engine.search("introduction", top_n=20, genre="Computer Science")
    assert not result.empty and set(result["genre"]) == {"Computer Science"}


def test_available_only_hides_unavailable_books(engine, conn):
    availability = db.availability_df(conn)
    result = engine.search("learning", top_n=20, available_only=True, availability=availability)
    assert not result.empty and (result["available"] > 0).all()


def test_results_are_sorted_best_first(engine):
    scores = engine.search("history of africa", top_n=10)["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_benchmark_hit_rate(books):
    result = evaluate_search(books, k=3)
    assert result["hit_rate"] >= 0.9, result["misses"]