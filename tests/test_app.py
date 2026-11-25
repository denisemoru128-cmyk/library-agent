"""Screen-level tests: the real Streamlit app is run against a private copy of the database.

Streamlit's AppTest clicks buttons and types into boxes like a user would, so these check that
the screens are wired to the rules (circulation), the agent and the chatbot, not just that
those modules work on their own.
"""
from library_agent import db, report


def _button(at, label_start):
    return next(b for b in at.button if b.label.startswith(label_start))


def _count(at, table):
    conn = db.connect(at.db)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
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

    at.session_state["user"] = {"role": "student", "student_id": 1}
    at.run()
    at.text_input(key="top_search").set_value("beekeeping").run()

    assert at.session_state["section"] == "Search"
    assert "Orchard Beekeeping Basics" in " ".join(h.value for h in at.get("html"))


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