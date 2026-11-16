import pytest

from library_agent import db
from library_agent.chatbot import LibraryChatbot
from library_agent.forecast import DemandForecaster
from library_agent.recommender import Recommender
from library_agent.search import BookSearch
from tests.conftest import AS_OF


@pytest.fixture(scope="module")
def chatbot(conn):
    books = db.books_df(conn)
    students = db.students_df(conn)
    loans = db.loans_df(conn)
    search = BookSearch(books)
    recommender = Recommender(books, students, loans, AS_OF)
    forecaster = DemandForecaster(books, loans, AS_OF)
    return LibraryChatbot(books, search, recommender, forecaster)


def test_classifier_recognizes_due_this_week(chatbot):
    intent, confidence = chatbot.classify("What books are due this week?")

    assert intent == "due_soon"
    assert confidence >= 0.45


def test_entity_extraction_finds_catalog_title(chatbot, books):
    title = books.iloc[0]["title"]

    assert chatbot.extract_entities(f'Is "{title}" available?')["book_title"] == title
    assert chatbot.extract_entities("Show recommendations for student 7")["student_id"] == 7


def test_availability_answer_requires_a_detected_title(chatbot, conn):
    title = db.books_df(conn).iloc[0]["title"]
    result = chatbot.answer(f'Is "{title}" available?', conn)

    assert result["intent"] == "availability"
    assert result["data"][0]["title"] == title
    assert "available" in result["data"][0]


def test_author_query_lists_available_books_by_partial_name(chatbot, conn):
    result = chatbot.answer("Which books are available from author thiongo?", conn)

    assert result["intent"] == "author_search"
    assert result["entities"]["author"] == "Ngugi wa Thiong'o"
    assert result["data"]
    assert all(book["author"] == "Ngugi wa Thiong'o" for book in result["data"])
    assert all(book["available"] > 0 for book in result["data"])


def test_recommendation_requires_a_selected_student(chatbot, conn):
    result = chatbot.answer("What should I read next?", conn)

    assert result["intent"] == "recommendations"
    assert "Select a student" in result["message"]
