import pandas as pd
import pytest

from library_agent import db
from library_agent.forecast import DemandForecaster
from tests.conftest import AS_OF


@pytest.fixture(scope="module")
def forecaster(conn):
    return DemandForecaster(db.books_df(conn), db.loans_df(conn), AS_OF)


def test_book_forecast_has_nonnegative_values_for_every_requested_week(forecaster, books):
    result = forecaster.forecast(by="book", horizon_weeks=3)

    assert len(result) == len(books) * 3
    assert result["week_start"].nunique() == 3
    assert (result["predicted_loans"] >= 0).all()
    assert result.groupby("week_start")["book_id"].nunique().eq(len(books)).all()


def test_genre_forecast_is_the_sum_of_book_forecasts(forecaster):
    by_book = forecaster.forecast(by="book", horizon_weeks=2)
    expected = (
        by_book.groupby(["week_start", "genre"], as_index=False)["predicted_loans"]
        .sum()
        .sort_values(["week_start", "genre"])
        .reset_index(drop=True)
    )
    actual = forecaster.forecast(by="genre", horizon_weeks=2).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected)


def test_forecast_rejects_invalid_horizon(forecaster):
    with pytest.raises(ValueError, match="horizon_weeks"):
        forecaster.forecast(horizon_weeks=0)
