"""Screen-level tests: the real Streamlit app is run against a private copy of the database.

Streamlit's AppTest clicks buttons and types into boxes like a user would, so these check that
the screens are wired to the rules (circulation), the agent and the chatbot, not just that
those modules work on their own.
"""
import copy
from datetime import date

from library_agent import db, report, stability


def _button(at, label_start):
    return next(b for b in at.button if b.label.startswith(label_start))


# ---- librarian: catalogue and model quality (the other librarian screens are in test_librarian_screens.py)
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

    page = " ".join(h.value for h in at.get("html"))
    assert "Search hit rate @3" in page and "Recommender hit rate @10" in page
    assert any("Evaluation results" in m.value for m in at.markdown)


def test_model_quality_section_says_how_to_create_the_results(app, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path / "nothing_here")
    at = app(role="Librarian", section="Model quality")

    assert "python evaluate.py" in " ".join(h.value for h in at.get("html"))


def test_model_quality_section_shows_the_range_across_libraries(app, quick_results, tmp_path, monkeypatch):
    several = copy.deepcopy(quick_results)
    several["stability"] = stability.summarise([quick_results, quick_results],
                                               stability.library_settings(42, date(2026, 11, 25), 2))
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path)
    report.save_results(several)
    at = app(role="Librarian", section="Model quality")

    assert any(c.value.startswith("Across 2 simulated libraries") for c in at.caption)
    assert any("How stable are these numbers?" in m.value for m in at.markdown)


def test_model_quality_section_suggests_the_multi_library_run_when_there_is_only_one(app, quick_results, tmp_path,
                                                                                    monkeypatch):
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path)
    report.save_results(quick_results)
    at = app(role="Librarian", section="Model quality")

    assert any("--libraries 5" in c.value for c in at.caption)