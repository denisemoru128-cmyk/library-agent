"""Shared test setup: one simulated database, built once, with a fixed 'today'."""
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library_agent import config, db  # noqa: E402
from library_agent.simulation import seed_database  # noqa: E402

AS_OF = date(2026, 11, 16)


def student_with_late_and_ok_loan(conn):
    """(student_id, late_loan_id, ok_loan_id, ok_title): exactly one overdue book, plus a renewable loan
    that is not overdue and has a plain title that can be quoted in a chat message."""
    from library_agent import circulation
    loans = circulation.open_loans(conn)
    for student_id, group in loans.groupby("student_id"):
        late = group[group["days_left"] < 0]
        ok = group[(group["days_left"] >= 0) & (group["renewals"] == 0) & ~group["title"].str.contains("['\",:]")]
        if len(late) == 1 and len(ok):
            return int(student_id), int(late.iloc[0]["loan_id"]), int(ok.iloc[0]["loan_id"]), str(ok.iloc[0]["title"])
    raise AssertionError("the simulated data should contain such a student")


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "test_library.db"
    seed_database(path, as_of=AS_OF)
    return path


@pytest.fixture(scope="session")
def conn(db_path):
    """Read-only use: shared by the whole test session."""
    connection = db.connect(db_path)
    yield connection
    connection.close()


@pytest.fixture()
def fresh_conn(db_path, tmp_path):
    """A private copy of the database for tests that change data."""
    copy = tmp_path / "copy.db"
    shutil.copy(db_path, copy)
    connection = db.connect(copy)
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def books(conn):
    return db.books_df(conn)


@pytest.fixture(scope="session")
def forecaster(conn):
    """One trained demand forecaster shared by every test that needs it (training takes a few seconds)."""
    from library_agent.forecast import DemandForecaster
    return DemandForecaster(db.books_df(conn), db.loans_df(conn), AS_OF)


@pytest.fixture(scope="session")
def quick_results(db_path):
    """The real evaluation (quick mode, about 40 s), run once and shared by the tests that read its results."""
    import evaluate
    return evaluate.collect(quick=True, database=db_path)


APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture()
def app(db_path, tmp_path_factory, monkeypatch):
    """Returns open_app(): starts the real Streamlit app, already signed in, on a fresh copy of the test database.

    The copy always has the same path, so the slow demand model is trained once for the whole
    test run (Streamlit caches it by path) instead of once per test. The sign-in screen has its own tests.
    """
    folder = tmp_path_factory.getbasetemp() / "app_data"
    folder.mkdir(exist_ok=True)
    copy = folder / "app.db"
    shutil.copy(db_path, copy)
    monkeypatch.setattr(config, "DB_PATH", copy)

    def open_app(role="Student", student_id=1, section=None):
        at = AppTest.from_file(APP_FILE, default_timeout=120)
        at.session_state["user"] = (
            {"role": "librarian"} if role == "Librarian" else {"role": "student", "student_id": int(student_id)})
        if section:
            at.session_state["section"] = section
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        at.db = copy
        return at

    return open_app