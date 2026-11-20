"""Weekly borrowing-demand forecasts using a pooled random-forest regressor,
plus acquisition advice ("how many copies will we need?") built on the forecast.

The forecast itself (class DemandForecaster.forecast) predicts loans per week
for every book from its own recent history, the academic calendar (exam weeks,
breaks) and its genre. ``book_demand`` turns those predictions into a decision:

    one copy can serve about  (days in the horizon / average loan length)  loans.
    We aim to keep copies at most 70% busy so students rarely have to wait.
    copies needed = forecast loans / (loans one copy can serve x 0.7)
    if copies needed > copies owned  ->  advise buying the difference.
"""
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

from .academic_calendar import phase
from .config import FORECAST_HORIZON_WEEKS, RANDOM_SEED

LAG_WEEKS = 8
MAX_HORIZON_WEEKS = 52
TARGET_UTILISATION = 0.7   # keep copies at most ~70% busy so students rarely wait
RECENT_WEEKS = 8           # window used to report "recent loans" next to the forecast


def copies_needed(forecast_loans: float, loans_per_copy: float) -> int:
    """Copies required to serve the forecast demand at the target utilisation (at least 1)."""
    return max(1, math.ceil(forecast_loans / (loans_per_copy * TARGET_UTILISATION)))


def advice_for(copies: int, needed: int, demand_per_copy: float, loans_per_copy: float) -> str:
    """Turn the numbers into a plain-language recommendation for the librarian."""
    gap = needed - copies
    if gap >= 1:
        return f"Buy {gap} more cop{'y' if gap == 1 else 'ies'}"
    # exactly enough copies, but they would be kept more than half busy: worth watching
    if gap == 0 and demand_per_copy >= loans_per_copy * 0.5:
        return "Watch"
    return "OK"


class DemandForecaster:
    """Forecast demand for each title, then optionally aggregate it by genre."""

    def __init__(self, books: pd.DataFrame, loans: pd.DataFrame, as_of: date):
        self.books = books.reset_index(drop=True).copy()
        self.as_of = pd.Timestamp(as_of).normalize()
        self._book_by_id = self.books.set_index("book_id")
        self._weekly = self._weekly_counts(loans)
        returned = loans.dropna(subset=["returned_on"])
        self.average_loan_days = (
            float((returned["returned_on"] - returned["borrowed_on"]).dt.days.mean())
            if len(returned) else 14.0
        )
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

            # Books are independent within one week, so predict them all in ONE model call
            # (400 single-row calls took 2.5 s; 4 batch calls are instant). The weeks themselves
            # must stay in order: each week's prediction becomes the next week's "last week".
            rows, histories, windows = [], [], []
            for row in self.books.itertuples(index=False):
                known = series_by_book.get(int(row.book_id), np.zeros(0))
                if len(known) < LAG_WEEKS:
                    values = np.pad(known, (max(0, LAG_WEEKS - len(known)), 0))[-LAG_WEEKS:]
                else:
                    values = known[-LAG_WEEKS:].copy()
                rows.append(row)
                histories.append(known)
                windows.append(values)

            if self._model is None:
                predicted = [float(values.mean()) for values in windows]
            else:
                features = [
                    self._features(int(row.book_id), row.genre, values, target_week)
                    for row, values in zip(rows, windows)
                ]
                predicted = self._model.predict(self._vectorizer.transform(features))

            for row, known, prediction in zip(rows, histories, predicted):
                book_id = int(row.book_id)
                prediction = round(max(float(prediction), 0.0), 3)
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

    # ------------------------------------------------------------ acquisition advice
    def book_demand(self, horizon_weeks: int = FORECAST_HORIZON_WEEKS,
                    availability: pd.DataFrame | None = None) -> pd.DataFrame:
        """One row per book: recent loans, forecast loans, copies needed and advice.

        Sorted so the books under the most pressure (forecast loans per copy) come first.
        """
        predicted = (
            self.forecast(by="book", horizon_weeks=horizon_weeks)
            .groupby("book_id")["predicted_loans"].sum()
        )
        recent = self._weekly.iloc[:, -RECENT_WEEKS:].sum(axis=1) if self._weekly.shape[1] else pd.Series(dtype=float)

        df = self.books[["book_id", "title", "author", "genre", "copies"]].copy()
        df["recent_loans"] = df["book_id"].map(recent).fillna(0).astype(int)
        df["forecast_loans"] = df["book_id"].map(predicted).fillna(0.0).round(1)

        loans_per_copy = max(horizon_weeks * 7 / max(self.average_loan_days, 1.0), 1.0)
        df["copies_needed"] = [copies_needed(f, loans_per_copy) for f in df["forecast_loans"]]
        df["demand_per_copy"] = (df["forecast_loans"] / df["copies"].clip(lower=1)).round(2)
        df["advice"] = [
            advice_for(c, n, d, loans_per_copy)
            for c, n, d in zip(df["copies"], df["copies_needed"], df["demand_per_copy"])
        ]
        if availability is not None:
            df = df.merge(availability[["book_id", "available"]], on="book_id", how="left")
        return df.sort_values(["demand_per_copy", "forecast_loans"], ascending=False).reset_index(drop=True)

    def high_demand_books(self, top_n: int = 10, horizon_weeks: int = FORECAST_HORIZON_WEEKS,
                          availability: pd.DataFrame | None = None) -> pd.DataFrame:
        """The librarian's 'what is under pressure' list."""
        return self.book_demand(horizon_weeks, availability).head(top_n)

    def total_series(self, horizon_weeks: int = FORECAST_HORIZON_WEEKS) -> pd.DataFrame:
        """Weekly total loans: history ('actual') followed by the forecast, for a chart."""
        history = self._weekly.sum(axis=0)
        actual = pd.DataFrame({"week_start": history.index.date, "loans": history.to_numpy(), "kind": "actual"})
        future = self.forecast(by="genre", horizon_weeks=horizon_weeks).groupby("week_start", as_index=False)[
            "predicted_loans"].sum()
        future = future.rename(columns={"predicted_loans": "loans"}).assign(kind="forecast")
        return pd.concat([actual, future], ignore_index=True)