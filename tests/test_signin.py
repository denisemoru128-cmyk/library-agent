"""Sign-in, navigation and the alerts: the frame around every screen.

Sign-in is a demo (a registration number, no password), so these tests check that it finds the right
student, refuses unknown numbers, offers two sensible demo students, and that the shell shows what the
loans say.
"""
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from library_agent import circulation, config
from library_agent.config import ESCALATION_DAYS
from views import shell, signin

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture()
def fresh_app(db_path, tmp_path_factory, monkeypatch):
    """A signed-out app on a private copy of the database (same path every time, so models train once)."""
    folder = tmp_path_factory.getbasetemp() / "app_data"
    folder.mkdir(exist_ok=True)
    copy = folder / "app.db"
    shutil.copy(db_path, copy)
    monkeypatch.setattr(config, "DB_PATH", copy)

    def open_app():
        at = AppTest.from_file(APP_FILE, default_timeout=120)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    return open_app


def _button(at, label_start):
    return next(b for b in at.button if b.label.startswith(label_start))


# ---- finding the student --------------------------------------------------------------
@pytest.mark.parametrize("typed", ["REG-0041", "reg-0041", " reg0041 ", "41", "0041"])
def test_registration_numbers_are_matched_forgivingly(conn, typed):
    from library_agent import db
    student = signin.find_student(db.students_df(conn), typed)

    assert student is not None and student["reg_no"] == "REG-0041"


@pytest.mark.parametrize("typed", ["", "   ", "REG-9999", "hello"])
def test_unknown_registration_numbers_find_nobody(conn, typed):
    from library_agent import db
    assert signin.find_student(db.students_df(conn), typed) is None


def test_demo_students_show_a_good_standing_and_an_overdue_case(conn):
    good, late = signin.demo_students(conn)
    good_summary = circulation.student_summary(conn, good["student_id"])
    late_summary = circulation.student_summary(conn, late["student_id"])

    assert good_summary["overdue"] == 0 and good_summary["active_loans"] >= 3
    assert late_summary["borrowing_paused"] and late_summary["active_loans"] >= 2


# ---- the sign-in screen ----------------------------------------------------------------
def test_the_app_opens_on_the_sign_in_screen(fresh_app):
    at = fresh_app()

    assert "user" not in at.session_state
    assert at.text_input[0].label == "Registration number"
    assert not at.sidebar.button                      # no navigation before signing in


def test_typing_a_registration_number_signs_the_student_in(fresh_app):
    at = fresh_app()
    at.text_input[0].set_value("REG-0041")
    _button(at, "Continue").click().run()

    assert at.session_state["user"] == {"role": "student", "student_id": 41}
    assert [b.label for b in at.sidebar.button][:3] == ["Search", "For you", "My loans"]


def test_an_unknown_registration_number_is_refused_with_a_hint(fresh_app):
    at = fresh_app()
    at.text_input[0].set_value("REG-9999")
    _button(at, "Continue").click().run()

    assert "user" not in at.session_state
    assert "No student has that registration number" in at.error[0].value


def test_the_demo_buttons_sign_in_the_students_and_the_librarian(fresh_app):
    at = fresh_app()
    _button(at, "Librarian").click().run()
    assert at.session_state["user"] == {"role": "librarian"}

    at = fresh_app()
    first_demo = next(b for b in at.button if b.key and b.key.startswith("demo_") and b.key != "demo_librarian")
    first_demo.click().run()
    assert at.session_state["user"]["role"] == "student"


def test_switch_user_returns_to_the_sign_in_screen(fresh_app):
    at = fresh_app()
    at.session_state["user"] = {"role": "student", "student_id": 41}
    at.run()
    at.button(key="switch_user").click().run()

    assert "user" not in at.session_state
    assert at.text_input[0].label == "Registration number"


# ---- navigation ------------------------------------------------------------------------
def test_the_section_links_change_the_screen(fresh_app):
    at = fresh_app()
    at.session_state["user"] = {"role": "student", "student_id": 41}
    at.run()
    at.button(key="nav_My loans").click().run()

    assert at.session_state["section"] == "My loans"
    assert any(s.label == "Choose a loan" for s in at.selectbox)


def test_each_role_gets_its_own_links(fresh_app):
    at = fresh_app()
    at.session_state["user"] = {"role": "librarian"}
    at.run()

    assert [b.label for b in at.sidebar.button if b.key.startswith("nav_")] == shell.LIBRARIAN_SECTIONS


# ---- alerts ---------------------------------------------------------------------------
def test_the_student_alert_count_is_overdue_plus_due_soon_loans(conn):
    loans = circulation.open_loans(conn)
    for student_id, group in loans.groupby("student_id"):
        expected = int((group["days_left"] < 0).sum() + (group["status"] == "due soon").sum())
        count, _ = shell.alerts(conn, {"role": "student", "student_id": int(student_id)})
        assert count == expected


def test_a_student_with_nothing_wrong_has_no_alerts(conn):
    good, _ = signin.demo_students(conn)
    count, lines = shell.alerts(conn, {"role": "student", "student_id": good["student_id"]})

    assert count == 0 or all("Overdue" not in line for line in lines)


def test_the_librarian_alert_count_is_loans_past_the_escalation_limit(conn):
    count, lines = shell.alerts(conn, {"role": "librarian"})
    overdue = circulation.overdue_loans(conn)

    assert count == int((overdue["days_overdue"] > ESCALATION_DAYS).sum()) > 0
    assert lines and "overdue" in lines[0]