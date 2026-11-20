import pandas as pd
import pytest

from library_agent import db
from library_agent.evaluation import evaluate_forecast
from library_agent.forecast import DemandForecaster, advice_for, copies_needed
from tests.conftest import AS_OF


@pytest.fixture(scope="module")
def forecaster(conn):
    return DemandForecaster(db.books_df(conn), db.loans_df(conn), AS_OF)


# ---- the decision rule on its own (no model involved) ----------------------
def test_copies_needed_rounds_up_and_is_at_least_one():
    # one copy serves 2.4 loans per 4 weeks; at 70% target that is 1.68 loans per copy
    assert copies_needed(0.0, 2.4) == 1
    assert copies_needed(1.6, 2.4) == 1
    assert copies_needed(1.7, 2.4) == 2
    assert copies_needed(5.0, 2.4) == 3


@pytest.mark.parametrize("copies, needed, per_copy_demand, expected", [
    (2, 4, 3.0, "Buy 2 more copies"),
    (2, 3, 2.0, "Buy 1 more copy"),
    (3, 3, 1.5, "Watch"),     # exactly enough, but kept more than half busy
    (3, 3, 0.4, "OK"),        # exactly enough, plenty of room
    (4, 2, 0.5, "OK"),        # more than enough
])
def test_advice_wording(copies, needed, per_copy_demand, expected):
    assert advice_for(copies, needed, per_copy_demand, loans_per_copy=2.4) == expected


# ---- the book-level table built from the forecast -------------------------
def test_book_demand_has_one_row_per_book_with_consistent_numbers(forecaster, books, conn):
    table = forecaster.book_demand(availability=db.availability_df(conn))
    assert len(table) == len(books) and table["book_id"].is_unique
    assert (table["forecast_loans"] >= 0).all() and (table["copies_needed"] >= 1).all()
    # the table must carry exactly what the forecast says (allowing for rounding to 1 decimal)
    total = forecaster.forecast(by="book", horizon_weeks=4)["predicted_loans"].sum()
    assert table["forecast_loans"].sum() == pytest.approx(total, abs=0.05 * len(books))
    # a book we plan to buy copies for must really be short of copies
    short = table[table["advice"].str.startswith("Buy")]
    assert (short["copies_needed"] > short["copies"]).all()
    assert "available" in table.columns


def test_high_demand_list_is_sorted_by_pressure(forecaster):
    top = forecaster.high_demand_books(top_n=5)
    assert len(top) == 5 and top["demand_per_copy"].is_monotonic_decreasing


def test_total_series_joins_history_and_forecast(forecaster):
    series = forecaster.total_series(horizon_weeks=4)
    assert (series["kind"] == "forecast").sum() == 4
    assert (series["kind"] == "actual").sum() > 50
    assert series["loans"].ge(0).all()


# ---- does the forecast actually beat doing nothing clever? ---------------
def test_backtest_beats_simple_baselines(conn):
    result = evaluate_forecast(db.books_df(conn), db.loans_df(conn), AS_OF, origins=2)
    assert result["genre"]["model"]["wape"] < result["genre"]["4-week average"]["wape"]
    assert result["book"]["model"]["wape"] < result["book"]["last 4 weeks repeated"]["wape"]