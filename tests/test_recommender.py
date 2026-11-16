import pytest

from library_agent import db
from library_agent.evaluation import evaluate_recommender
from library_agent.recommender import Recommender
from tests.conftest import AS_OF


@pytest.fixture(scope="module")
def model(conn):
    return Recommender(db.books_df(conn), db.students_df(conn), db.loans_df(conn), AS_OF)


def test_returns_requested_number_sorted_by_score(model):
    recs = model.recommend(7, top_n=6)
    assert len(recs) == 6
    assert recs["score"].is_monotonic_decreasing


def test_never_recommends_a_book_already_borrowed(model, books):
    borrowed_ids = {int(books.at[pos, "book_id"]) for pos in model.history(7)}
    recs = model.recommend(7, top_n=30)
    assert not borrowed_ids & set(recs["book_id"])


def test_every_recommendation_has_a_reason(model):
    assert model.recommend(7, top_n=5)["reason"].str.len().gt(0).all()


def test_only_available_filter(model, conn):
    availability = db.availability_df(conn)
    recs = model.recommend(7, top_n=10, availability=availability, only_available=True)
    assert not recs.empty and (recs["available"] > 0).all()


def test_unknown_method_is_rejected(model):
    with pytest.raises(ValueError):
        model.recommend(7, method="magic")


def test_similar_books_excludes_the_book_itself(model):
    similar = model.similar_books(1, top_n=5)
    assert len(similar) == 5 and 1 not in set(similar["book_id"])


def test_new_student_gets_recommendations_from_declared_interests(fresh_conn):
    """Cold start: no loans at all, so the interests and popularity must drive the result."""
    fresh_conn.execute(
        "INSERT INTO students (student_id, reg_no, name, department, interests) "
        "VALUES (9999, 'REG-9999', 'New Student', 'Law', 'Law & Governance')"
    )
    fresh_conn.commit()
    model = Recommender(db.books_df(fresh_conn), db.students_df(fresh_conn), db.loans_df(fresh_conn), AS_OF)
    recs = model.recommend(9999, top_n=5)
    assert (recs["genre"] == "Law & Governance").sum() >= 4
    assert "Law & Governance" in recs.iloc[0]["reason"]


def test_hybrid_beats_baselines_on_future_loans(conn):
    result = evaluate_recommender(db.books_df(conn), db.students_df(conn), db.loans_df(conn), AS_OF)
    m = result["methods"]
    assert m["hybrid"]["hit_rate"] > m["random"]["hit_rate"] + 0.15
    assert m["hybrid"]["hit_rate"] >= m["popular"]["hit_rate"]
    assert m["hybrid"]["recall"] >= m["popular"]["recall"]


def test_hybrid_helps_most_for_students_with_little_history(conn):
    result = evaluate_recommender(db.books_df(conn), db.students_df(conn), db.loans_df(conn), AS_OF, max_history=3)
    m = result["methods"]
    assert m["hybrid"]["hit_rate"] > m["popular"]["hit_rate"] + 0.05