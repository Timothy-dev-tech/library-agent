from datetime import timedelta

import pytest

from library_agent import db
from tests.conftest import AS_OF


def test_due_and_overdue_queries_only_return_open_matching_loans(conn):
    overdue = db.overdue_loans_df(conn, as_of=AS_OF)
    due_soon = db.due_loans_df(conn, start=AS_OF, end=AS_OF + timedelta(days=6))

    assert not overdue.empty
    assert (overdue["due_on"].dt.date < AS_OF).all()
    assert (overdue["days_overdue"] > 0).all()
    assert (due_soon["due_on"].dt.date >= AS_OF).all()
    assert (due_soon["due_on"].dt.date <= AS_OF + timedelta(days=6)).all()


def test_borrow_and_return_update_availability(fresh_conn):
    fresh_conn.execute(
        "INSERT INTO students (student_id, reg_no, name, department) "
        "VALUES (9999, 'REG-9999', 'Test Student', 'Law')"
    )
    fresh_conn.commit()
    before = db.availability_df(fresh_conn)
    book_id = int(before.loc[before["available"] > 0, "book_id"].iloc[0])

    loan_id = db.borrow_book(fresh_conn, 9999, book_id, borrowed_on=AS_OF)
    after_borrow = db.availability_df(fresh_conn).set_index("book_id").at[book_id, "available"]
    assert after_borrow == before.set_index("book_id").at[book_id, "available"] - 1

    db.record_return(fresh_conn, loan_id, returned_on=AS_OF + timedelta(days=1))
    after_return = db.availability_df(fresh_conn).set_index("book_id").at[book_id, "available"]
    assert after_return == before.set_index("book_id").at[book_id, "available"]
    with pytest.raises(ValueError, match="already been returned"):
        db.record_return(fresh_conn, loan_id, returned_on=AS_OF + timedelta(days=1))


def test_borrow_rejects_a_book_without_available_copies(fresh_conn):
    fresh_conn.execute(
        "INSERT INTO students (student_id, reg_no, name, department) "
        "VALUES (9999, 'REG-9999', 'Test Student', 'Law')"
    )
    fresh_conn.commit()
    book_id = db.add_book(fresh_conn, "Unavailable title", "Test Author", "Test", "No copies", 0)

    with pytest.raises(ValueError, match="No copies"):
        db.borrow_book(fresh_conn, 9999, book_id, borrowed_on=AS_OF)
