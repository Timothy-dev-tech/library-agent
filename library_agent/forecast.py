"""Weekly borrowing-demand forecasts using a pooled random-forest regressor."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

from .academic_calendar import phase
from .config import FORECAST_HORIZON_WEEKS, RANDOM_SEED

LAG_WEEKS = 8
MAX_HORIZON_WEEKS = 52


class DemandForecaster:
    """Forecast demand for each title, then optionally aggregate it by genre."""

    def __init__(self, books: pd.DataFrame, loans: pd.DataFrame, as_of: date):
        self.books = books.reset_index(drop=True).copy()
        self.as_of = pd.Timestamp(as_of).normalize()
        self._book_by_id = self.books.set_index("book_id")
        self._weekly = self._weekly_counts(loans)
        self._vectorizer = DictVectorizer()
        self._model = None
        self._fit()

    def _weekly_counts(self, loans: pd.DataFrame) -> pd.DataFrame:
        ids = self.books["book_id"].tolist()
        as_of_week = self.as_of - pd.Timedelta(days=self.as_of.weekday())
        last_complete_week = (
            as_of_week if self.as_of.weekday() == 6
            else as_of_week - pd.Timedelta(days=7)
        )
        history = loans.copy()
        if history.empty:
            return pd.DataFrame(index=ids)

        history["borrowed_on"] = pd.to_datetime(history["borrowed_on"])
        history = history[history["borrowed_on"] <= self.as_of]
        if history.empty:
            return pd.DataFrame(index=ids)

        history["week_start"] = (
            history["borrowed_on"].dt.normalize()
            - pd.to_timedelta(history["borrowed_on"].dt.day_of_week, unit="D")
        )
        first_week = history["week_start"].min()
        weeks = pd.date_range(first_week, last_complete_week, freq="W-MON")
        if len(weeks) == 0:
            return pd.DataFrame(index=ids)

        counts = history[history["week_start"] <= last_complete_week].groupby(
            ["book_id", "week_start"]
        ).size()
        index = pd.MultiIndex.from_product([ids, weeks], names=["book_id", "week_start"])
        return counts.reindex(index, fill_value=0).unstack("week_start").astype(float)

    @staticmethod
    def _features(book_id: int, genre: str, values: np.ndarray, target_week: pd.Timestamp) -> dict:
        calendar = phase(target_week.date())
        features = {
            f"book={book_id}": 1.0,
            f"genre={genre}": 1.0,
            "lag_1": float(values[-1]),
            "lag_2": float(values[-2]),
            "lag_4": float(values[-4]),
            "lag_8": float(values[-8]),
            "mean_4": float(values[-4:].mean()),
            "mean_8": float(values[-8:].mean()),
            "week_of_term": float(calendar.week_of_term),
            "is_exam": float(calendar.is_exam),
            "is_break": float(calendar.is_break),
            "week_sin": float(np.sin(2 * np.pi * target_week.isocalendar().week / 52)),
            "week_cos": float(np.cos(2 * np.pi * target_week.isocalendar().week / 52)),
        }
        return features

    def _fit(self) -> None:
        if self._weekly.shape[1] <= LAG_WEEKS:
            return

        feature_rows, targets = [], []
        weeks = list(self._weekly.columns)
        genres = self._book_by_id["genre"]
        for book_id, series in self._weekly.iterrows():
            values = series.to_numpy(dtype=float)
            for index in range(LAG_WEEKS, len(weeks)):
                feature_rows.append(
                    self._features(book_id, genres.loc[book_id], values[:index], weeks[index])
                )
                targets.append(values[index])

        if not targets:
            return
        matrix = self._vectorizer.fit_transform(feature_rows)
        self._model = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        self._model.fit(matrix, np.asarray(targets))

    def forecast(
        self,
        by: str = "genre",
        horizon_weeks: int = FORECAST_HORIZON_WEEKS,
    ) -> pd.DataFrame:
        """Return expected weekly loans for each genre or individual book."""
        if by not in {"genre", "book"}:
            raise ValueError("by must be 'genre' or 'book'")
        if not 1 <= horizon_weeks <= MAX_HORIZON_WEEKS:
            raise ValueError(f"horizon_weeks must be between 1 and {MAX_HORIZON_WEEKS}")

        weekly_predictions = []
        last_week = self._weekly.columns[-1] if len(self._weekly.columns) else None
        series_by_book = {
            int(book_id): series.to_numpy(dtype=float)
            for book_id, series in self._weekly.iterrows()
        }
        for offset in range(1, horizon_weeks + 1):
            if last_week is None:
                target_week = self.as_of - pd.Timedelta(days=self.as_of.weekday())
                target_week += pd.Timedelta(weeks=offset)
            else:
                target_week = last_week + pd.Timedelta(weeks=offset)

            for row in self.books.itertuples(index=False):
                book_id = int(row.book_id)
                known = series_by_book.get(book_id, np.zeros(0))
                if len(known) < LAG_WEEKS:
                    values = np.pad(known, (max(0, LAG_WEEKS - len(known)), 0))[-LAG_WEEKS:]
                else:
                    values = known[-LAG_WEEKS:].copy()

                if self._model is None:
                    prediction = float(values.mean())
                else:
                    features = self._features(book_id, row.genre, values, target_week)
                    prediction = float(self._model.predict(self._vectorizer.transform([features]))[0])
                prediction = round(max(prediction, 0.0), 3)
                weekly_predictions.append({
                    "week_start": target_week.date(),
                    "book_id": book_id,
                    "title": row.title,
                    "genre": row.genre,
                    "predicted_loans": prediction,
                })
                series_by_book[book_id] = np.append(known, prediction)

        result = pd.DataFrame(weekly_predictions)
        if by == "genre":
            result = (
                result.groupby(["week_start", "genre"], as_index=False)["predicted_loans"]
                .sum()
                .sort_values(["week_start", "genre"])
            )
        else:
            result = result.sort_values(["week_start", "book_id"])
        return result.reset_index(drop=True)
