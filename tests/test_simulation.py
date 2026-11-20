"""The simulated history must obey the same rules the app enforces on real loans."""
import pandas as pd

from library_agent import db
from tests.conftest import AS_OF


def test_no_student_holds_two_copies_of_the_same_book_at_once(conn):
    loans = db.loans_df(conn).copy()
    loans["borrowed_on"] = pd.to_datetime(loans["borrowed_on"])
    loans["end"] = pd.to_datetime(loans["returned_on"]).fillna(pd.Timestamp(AS_OF))
    loans = loans.sort_values(["student_id", "book_id", "borrowed_on"])
    previous_end = loans.groupby(["student_id", "book_id"])["end"].shift()

    assert not (loans["borrowed_on"] < previous_end).any()


def test_nobody_exceeds_the_loan_limit_at_the_end_of_history(conn):
    from library_agent.config import MAX_ACTIVE_LOANS
    open_loans = db.loans_df(conn)
    open_loans = open_loans[open_loans["returned_on"].isna()]

    assert open_loans.groupby("student_id").size().max() <= MAX_ACTIVE_LOANS