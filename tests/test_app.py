"""Screen-level tests: the real Streamlit app is run against a private copy of the database.

Streamlit's AppTest clicks buttons and types into boxes like a user would, so these check that
the screens are wired to the rules (circulation), the agent and the chatbot, not just that
those modules work on their own.
"""
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from library_agent import circulation, config, db, report

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture()
def app(db_path, tmp_path_factory, monkeypatch):
    """Returns open_app(): starts the app on a fresh copy of the test database.

    The copy always has the same path, so the slow demand model is trained once for the whole
    test run (Streamlit caches it by path) instead of once per test.
    """
    folder = tmp_path_factory.getbasetemp() / "app_data"
    folder.mkdir(exist_ok=True)
    copy = folder / "app.db"
    shutil.copy(db_path, copy)
    monkeypatch.setattr(config, "DB_PATH", copy)

    def open_app(role="Student", student_id=None, section=None):
        at = AppTest.from_file(APP_FILE, default_timeout=120)
        at.run()
        if role == "Librarian":
            at.sidebar.radio[0].set_value("Librarian").run()
        elif student_id is not None:
            at.sidebar.selectbox[0].set_value(student_id).run()
        if section:
            at.segmented_control[0].set_value(section).run()
        assert not at.exception, [e.value for e in at.exception]
        at.db = copy
        return at

    return open_app


def _button(at, label_start):
    return next(b for b in at.button if b.label.startswith(label_start))


def _count(at, table):
    conn = db.connect(at.db)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def _first_loan(at, late: bool):
    """(student_id, loan_id) of a loan that is overdue, or one that is fine to renew."""
    conn = db.connect(at.db)
    try:
        loans = circulation.open_loans(conn)
    finally:
        conn.close()
    if late:
        row = loans[loans["days_left"] < 0].iloc[0]
    else:
        bad = set(loans.loc[loans["days_left"] < 0, "student_id"])
        row = loans[~loans["student_id"].isin(bad) & (loans["renewals"] == 0)].iloc[0]
    return int(row["student_id"]), int(row["loan_id"])


# ---- student screens ---------------------------------------------------------------
def test_student_screen_shows_the_loan_summary(app):
    at = app()

    assert [m.label for m in at.metric] == ["Books on loan", "Overdue", "Fines owed", "Can still borrow"]


def test_search_screen_finds_books_despite_typos(app):
    at = app(section="Search")
    at.text_input[0].set_value("machin lerning").run()

    assert not at.exception
    titles = " ".join(at.dataframe[0].value["title"])
    assert "Machine Learning" in titles


def test_chat_screen_remembers_the_book_between_messages(app):
    at = app(section="Ask the assistant")
    at.chat_input[0].set_value("find books about statistics").run()
    at.chat_input[0].set_value("is it available?").run()

    replies = [c.markdown[0].value for c in at.chat_message if c.name == "assistant"]
    assert replies[0].startswith("Found") and replies[1].startswith("Assuming you mean")


def test_renewing_from_the_loans_screen_updates_the_database(app):
    at = app(section="My loans")
    student_id, loan_id = _first_loan(at, late=False)
    at = app(student_id=student_id, section="My loans")
    next(s for s in at.selectbox if s.label == "Choose a loan").set_value(loan_id)
    _button(at, "Renew").click().run()

    assert at.success[0].value.startswith("Renewed")
    conn = db.connect(at.db)
    try:
        assert db.get_loan(conn, loan_id)["renewals"] == 1
    finally:
        conn.close()


def test_returning_a_late_book_charges_the_fine(app):
    at = app(section="My loans")
    student_id, loan_id = _first_loan(at, late=True)
    at = app(student_id=student_id, section="My loans")
    next(s for s in at.selectbox if s.label == "Choose a loan").set_value(loan_id)
    _button(at, "Return").click().run()

    assert "Fine: KES" in at.success[0].value
    conn = db.connect(at.db)
    try:
        assert db.get_loan(conn, loan_id)["returned_on"] is not None
    finally:
        conn.close()


# ---- librarian screens ------------------------------------------------------------------
def test_librarian_dashboard_loads_with_charts_and_tables(app):
    at = app(role="Librarian")

    assert [m.label for m in at.metric][:2] == ["Titles", "Copies on loan"]
    assert len(at.dataframe) >= 2


def test_preview_changes_nothing_and_running_the_agent_writes_a_log(app):
    at = app(role="Librarian", section="Agent")
    _button(at, "Preview").click().run()
    assert _count(at, "agent_log") == 0 and _count(at, "notifications") == 0

    _button(at, "Run agent now").click().run()
    assert not at.exception
    assert _count(at, "agent_log") > 0 and _count(at, "notifications") > 0


def test_running_the_agent_twice_in_a_day_sends_nothing_twice(app):
    at = app(role="Librarian", section="Agent")
    _button(at, "Run agent now").click().run()
    sent, logged = _count(at, "notifications"), _count(at, "agent_log")

    _button(at, "Run agent now").click().run()

    assert (_count(at, "notifications"), _count(at, "agent_log")) == (sent, logged)


def test_advancing_the_date_moves_the_library_clock(app):
    at = app(role="Librarian", section="Agent")
    _button(at, "Advance date").click().run()

    conn = db.connect(at.db)
    try:
        assert db.today(conn).isoformat() == "2026-11-17"
    finally:
        conn.close()


def test_a_new_book_is_searchable_straight_away(app):
    at = app(role="Librarian", section="Catalogue")
    at.text_input[0].set_value("Orchard Beekeeping Basics")
    at.text_input[1].set_value("A. Keeper")
    at.text_area[0].set_value("A practical guide to keeping bees in an orchard.")
    _button(at, "Add book").click().run()
    assert at.success[0].value.startswith("Added")

    at.sidebar.radio[0].set_value("Student").run()
    at.segmented_control[0].set_value("Search").run()
    at.text_input[0].set_value("beekeeping").run()

    assert "Orchard Beekeeping Basics" in set(at.dataframe[0].value["title"])


def test_adding_a_duplicate_title_is_refused(app):
    at = app(role="Librarian", section="Catalogue")
    existing = db.books_df(db.connect(at.db)).iloc[0]["title"]
    at.text_input[0].set_value(existing)
    at.text_input[1].set_value("Someone")
    at.text_area[0].set_value("Anything")
    _button(at, "Add book").click().run()

    assert "already in the catalogue" in at.error[0].value


def test_model_quality_section_shows_the_saved_results(app, quick_results, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path)
    report.save_results(quick_results)
    at = app(role="Librarian", section="Model quality")

    assert [m.label for m in at.metric][:2] == ["Search hit rate @3", "Recommender hit rate @10"]
    assert any("Evaluation results" in m.value for m in at.markdown)


def test_model_quality_section_says_how_to_create_the_results(app, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path / "nothing_here")
    at = app(role="Librarian", section="Model quality")

    assert any("python evaluate.py" in info.value for info in at.info)