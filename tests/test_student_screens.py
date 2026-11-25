"""The student screens: Home, Search, My loans and For you.

Two kinds of test. The first kind checks the buttons against the rules: for many students and books, a
button is switched off exactly when the library would refuse the action, and the hint says why. The second
kind runs the real app (Streamlit's AppTest) and clicks.
"""
import re
import shutil

import pytest

from library_agent import circulation, db
from library_agent.config import MAX_ACTIVE_LOANS
from views import components, signin


def _html(at) -> str:
    return " ".join(element.value for element in at.get("html"))


def _titles_shown(at) -> list[str]:
    return [t.replace("&amp;", "&").replace("&#x27;", "'") for t in re.findall(r'class="bt">(.*?)</div>', _html(at))]


def _private_copy(db_path, tmp_path, name):
    copy = tmp_path / f"{name}.db"
    shutil.copy(db_path, copy)
    return db.connect(copy)


# ---- the buttons agree with the rules ---------------------------------------------------
def _borrow_cases(conn):
    """One (student_id, book_id) pair for each way borrowing can be refused, plus one that is allowed."""
    loans = circulation.open_loans(conn)
    free = db.availability_df(conn).set_index("book_id")["available"]
    counts = loans.groupby("student_id").size()
    late = set(loans.loc[loans["days_left"] < 0, "student_id"])
    students = db.students_df(conn)["student_id"]
    free_book = int(free[free > 0].index[0])
    clean = [int(s) for s in students if s not in late and counts.get(s, 0) < MAX_ACTIVE_LOANS]
    holder = loans[~loans["student_id"].isin(late) & (loans["book_id"].map(free) > 0)].iloc[0]
    return {
        "no_copy": (int(students.iloc[0]), int(free[free == 0].index[0])),
        "limit": (int(counts[(counts == MAX_ACTIVE_LOANS - 1) & ~counts.index.isin(late)].index[0]), free_book),
        "overdue": (int(next(iter(late))), free_book),
        "has_it": (int(holder["student_id"]), int(holder["book_id"])),
        "allowed": (clean[0], next(int(b) for b in free[free > 0].index
                                   if b not in set(loans.loc[loans["student_id"] == clean[0], "book_id"]))),
    }


def _fill_to_the_limit(conn, student_id: int, keep_free: int) -> None:
    """Lend the student books until they hold the maximum (the seeded data has nobody at the limit)."""
    free = db.availability_df(conn)
    spare = [int(b) for b in free.loc[free["available"] > 0, "book_id"] if int(b) != keep_free]
    for book_id in spare:
        if len(circulation.open_loans(conn, student_id)) >= MAX_ACTIVE_LOANS:
            return
        circulation.issue_book(conn, student_id, book_id)


def test_borrow_button_is_off_exactly_when_the_library_would_refuse(conn, db_path, tmp_path):
    expected_hint = {"no_copy": "No free copy", "limit": "the limit", "overdue": "overdue book",
                     "has_it": "Already on your loans", "allowed": None}
    for kind, (student_id, book_id) in _borrow_cases(conn).items():
        private = _private_copy(db_path, tmp_path, f"borrow_{kind}")
        try:
            if kind == "limit":
                _fill_to_the_limit(private, student_id, keep_free=book_id)
            loans = circulation.open_loans(private, student_id)
            free = db.availability_df(private).set_index("book_id")["available"]
            hint = circulation.borrow_blocker(loans, book_id, int(free[book_id]))
            outcome = circulation.issue_book(private, student_id, book_id)
        finally:
            private.close()
        if expected_hint[kind] is None:
            assert hint is None, kind
        else:
            assert expected_hint[kind] in hint, (kind, hint)
        assert outcome.ok == (hint is None), f"{kind}: the button said {hint!r} but the library said {outcome.message!r}"


def _renew_cases(conn):
    loans = circulation.open_loans(conn)
    late_students = set(loans.loc[loans["days_left"] < 0, "student_id"])
    own_late = loans[loans["days_left"] < 0].iloc[0]
    blocked_by_other = loans[(loans["days_left"] >= 0) & loans["student_id"].isin(late_students)].iloc[0]
    renewed = loans[(loans["renewals"] >= 1) & ~loans["student_id"].isin(late_students)]
    allowed = loans[(loans["days_left"] >= 0) & (loans["renewals"] == 0) & ~loans["student_id"].isin(late_students)]
    cases = {"overdue": own_late, "other_overdue": blocked_by_other, "allowed": allowed.iloc[0]}
    if not renewed.empty:
        cases["max_renewals"] = renewed.iloc[0]
    return cases


def test_renew_button_is_off_exactly_when_the_library_would_refuse(conn, db_path, tmp_path):
    for kind, row in _renew_cases(conn).items():
        student_loans = circulation.open_loans(conn, int(row["student_id"]))
        hint = circulation.renew_blocker(student_loans, int(row["loan_id"]))

        private = _private_copy(db_path, tmp_path, f"renew_{kind}")
        try:
            outcome = circulation.renew_loan(private, int(row["loan_id"]))
        finally:
            private.close()
        assert outcome.ok == (hint is None), f"{kind}: the button said {hint!r} but the library said {outcome.message!r}"
        assert (kind == "allowed") == (hint is None), kind


def test_every_loan_gets_a_renew_answer_that_matches_the_rules(conn):
    """Whatever the data, a loan is renewable only when it is on time, nothing else is overdue and it has renewals left."""
    loans = circulation.open_loans(conn)
    late_by_student = loans[loans["days_left"] < 0].groupby("student_id").size()
    for row in loans.itertuples():
        student_loans = loans[loans["student_id"] == row.student_id]
        hint = circulation.renew_blocker(student_loans, int(row.loan_id))
        should_work = row.days_left >= 0 and row.student_id not in late_by_student.index and row.renewals < 1
        assert (hint is None) == should_work


# ---- the headline numbers ---------------------------------------------------------------
def test_the_stat_strip_matches_the_student_summary_for_every_student(conn):
    for student_id in db.students_df(conn)["student_id"]:
        summary = circulation.student_summary(conn, int(student_id))
        loans = circulation.open_loans(conn, int(student_id))
        on_loan, overdue, fines, fourth = components.student_tiles(summary, loans)

        assert on_loan["value"] == summary["active_loans"] and on_loan["unit"] == f"of {MAX_ACTIVE_LOANS}"
        assert overdue["value"] == summary["overdue"]
        assert fines["value"] == f"KES {summary['fines_owed']}"
        assert (fourth["label"] == "Borrowing") == summary["borrowing_paused"]
        assert ("late" in overdue.get("tone", "")) == (summary["overdue"] > 0)


def test_dates_are_written_with_the_right_weekday(conn):
    row = circulation.open_loans(conn).iloc[0]
    text, _ = components.due_line(next(circulation.open_loans(conn).itertuples()))

    assert text.startswith("Due ") and row["due_on"].strftime("%a") in text


def test_availability_wording(conn):
    assert components.availability_line(3) == ("3 copies free", "ok")
    assert components.availability_line(1) == ("1 copy free", "ok")
    assert components.availability_line(0, {"copies_out": 2, "next_due": None, "late_copies": 2})[0] == \
        "Both copies out · overdue, no return date"
    from datetime import date
    text, tone = components.availability_line(0, {"copies_out": 3, "next_due": date(2026, 11, 28), "late_copies": 0})
    assert text == "All 3 copies out · next back Sat 28 Nov" and tone == "soon"


# ---- Home -------------------------------------------------------------------------------
def test_home_greets_the_student_and_shows_the_strip(app, conn):
    good, _ = signin.demo_students(conn)
    at = app(student_id=good["student_id"])
    page = _html(at)

    assert "<small>On loan</small>" in page and "<small>Next due</small>" in page
    first_name = good["text"].split()[0]
    assert any(first_name in m.value and "Welcome back" in m.value for m in at.markdown)


def test_home_shows_at_most_three_loans_and_three_picks(app, conn):
    good, _ = signin.demo_students(conn)
    at = app(student_id=good["student_id"])
    keys = [b.key for b in at.button]

    assert 1 <= len([k for k in keys if k.startswith("home_renew_")]) <= 3
    assert len([k for k in keys if k.startswith("home_borrow_")]) == 3


def test_a_student_with_an_overdue_book_cannot_borrow_or_renew_and_is_told_why(app, conn):
    _, late = signin.demo_students(conn)
    at = app(student_id=late["student_id"])
    borrow = [b for b in at.button if b.key.startswith("home_borrow_")]
    renew = [b for b in at.button if b.key.startswith("home_renew_")]
    returns = [b for b in at.button if b.key.startswith("home_return_")]

    assert all(b.disabled for b in borrow + renew)
    assert returns and not any(b.disabled for b in returns)
    page = _html(at)
    assert "Return your overdue book first" in page and "Return the overdue book first" in page
    assert "Paused" in page


def test_renewing_from_home_updates_the_database(app, conn):
    good, _ = signin.demo_students(conn)
    student_id = good["student_id"]
    loans = circulation.open_loans(conn, student_id).sort_values("days_left").head(3)
    loan_id = int(loans[loans["renewals"] == 0].iloc[0]["loan_id"])

    at = app(student_id=student_id)
    at.button(key=f"home_renew_{loan_id}").click().run()

    assert not at.exception and at.success[0].value.startswith("Renewed")
    private = db.connect(at.db)
    try:
        assert db.get_loan(private, loan_id)["renewals"] == 1
    finally:
        private.close()
    assert at.button(key=f"home_renew_{loan_id}").disabled            # and now it says it was already renewed
    assert "Already renewed" in _html(at)


# ---- My loans ---------------------------------------------------------------------------
def test_my_loans_for_a_late_student_shows_the_banner_and_one_primary_return(app, conn):
    _, late = signin.demo_students(conn)
    student_id = late["student_id"]
    at = app(student_id=student_id, section="My loans")
    page = _html(at)
    loans = circulation.open_loans(conn, student_id)
    overdue_ids = {int(i) for i in loans.loc[loans["days_left"] < 0, "loan_id"]}

    assert 'class="banner"' in page and "overdue" in page and "can’t borrow or renew" in page
    returns = {int(b.key.split("_")[-1]): b for b in at.button if b.key.startswith("loans_return_")}
    assert set(returns) == set(map(int, loans["loan_id"]))            # every loan can be returned
    assert {i for i, b in returns.items() if b.proto.type == "primary"} == overdue_ids
    renewals = {int(b.key.split("_")[-1]) for b in at.button if b.key.startswith("loans_renew_")}
    assert renewals == set(map(int, loans["loan_id"])) - overdue_ids  # an overdue book has no Renew button
    assert all(at.button(key=f"loans_renew_{i}").disabled for i in renewals)  # the others wait for it


def test_returning_a_late_book_from_my_loans_charges_the_fine(app, conn):
    _, late = signin.demo_students(conn)
    loans = circulation.open_loans(conn, late["student_id"])
    row = loans[loans["days_left"] < 0].iloc[0]

    at = app(student_id=late["student_id"], section="My loans")
    at.button(key=f"loans_return_{int(row['loan_id'])}").click().run()

    assert f"Fine: KES {int(row['fine'])}" in at.success[0].value
    private = db.connect(at.db)
    try:
        assert db.get_loan(private, int(row["loan_id"]))["returned_on"] is not None
        paused = circulation.student_summary(private, late["student_id"])["borrowing_paused"]
    finally:
        private.close()
    assert ("Paused" in _html(at)) == paused                           # the screen follows the new situation


def test_my_loans_with_nothing_on_loan_offers_the_next_step(app, conn):
    loans = circulation.open_loans(conn)
    nobody = next(int(s) for s in db.students_df(conn)["student_id"] if s not in set(loans["student_id"]))
    at = app(student_id=nobody, section="My loans")

    assert "No books on loan" in _html(at)
    at.button(key="empty_search").click().run()
    assert at.session_state["section"] == "Search"


# ---- Search -----------------------------------------------------------------------------
def test_the_top_search_box_opens_search_and_finds_books_despite_typos(app):
    at = app()
    at.text_input(key="top_search").set_value("machin lerning").run()

    assert at.session_state["section"] == "Search"
    assert any("Machine Learning" in title for title in _titles_shown(at))
    assert "you typed" in " ".join(c.value for c in at.caption)         # says what it searched for instead


def test_search_without_a_query_offers_examples_and_they_work(app):
    at = app(section="Search")
    assert "Search the catalogue" in _html(at)

    at.button(key="try_statistics").click().run()
    assert at.session_state["search_query"] == "statistics"
    assert any("Statistic" in title for title in _titles_shown(at))


def test_search_can_hide_titles_with_no_free_copy(app, conn):
    at = app(section="Search")
    at.text_input(key="top_search").set_value("law").run()
    everything = _titles_shown(at)
    at.checkbox(key="student_available_only").check().run()
    free_only = _titles_shown(at)

    free = db.availability_df(conn).merge(db.books_df(conn)[["book_id", "title"]], on="book_id")
    out_titles = set(free.loc[free["available"] == 0, "title"])
    assert set(free_only) <= set(everything)
    assert not (set(free_only) & out_titles)


# ---- For you ----------------------------------------------------------------------------
def test_for_you_lists_picks_with_a_reason_and_borrowing_works(app, conn):
    good, _ = signin.demo_students(conn)
    student_id = good["student_id"]
    at = app(student_id=student_id, section="For you")
    page = _html(at)
    assert "Because you borrowed" in page

    button = next(b for b in at.button if b.key.startswith("reco_borrow_") and not b.disabled)
    before = circulation.student_summary(conn, student_id)["active_loans"]
    button.click().run()

    assert at.success[0].value.startswith("Issued")
    private = db.connect(at.db)
    try:
        assert circulation.student_summary(private, student_id)["active_loans"] == before + 1
    finally:
        private.close()


def test_for_you_can_be_limited_to_one_subject(app, conn):
    good, _ = signin.demo_students(conn)
    at = app(student_id=good["student_id"], section="For you")
    books = db.books_df(conn)
    for subject in ("Computer Science", "Business & Economics", "World Fiction"):
        at.selectbox(key="reco_genre").set_value(subject).run()
        shown = _titles_shown(at)
        genres = set(books.loc[books["title"].isin(shown), "genre"])
        assert genres <= {subject}, (subject, genres)
        if shown:
            break
    else:
        pytest.fail("expected at least one of these subjects to have a pick for the demo student")