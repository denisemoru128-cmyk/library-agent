from datetime import date, timedelta

from library_agent import circulation, db
from library_agent.config import FINE_PER_DAY, LOAN_DAYS, MAX_ACTIVE_LOANS
from tests.conftest import AS_OF


def _free_student(conn):
    """A student with no open loans (so the borrowing limit is not in the way)."""
    row = conn.execute(
        "SELECT student_id FROM students WHERE student_id NOT IN "
        "(SELECT student_id FROM loans WHERE returned_on IS NULL) LIMIT 1").fetchone()
    return row["student_id"]


def _free_book(conn):
    return conn.execute(
        "SELECT b.book_id FROM books b WHERE b.copies > (SELECT COUNT(*) FROM loans l "
        "WHERE l.book_id = b.book_id AND l.returned_on IS NULL) LIMIT 1").fetchone()["book_id"]


def test_fine_calculation():
    due = date(2026, 11, 1)
    assert circulation.fine_for(due, date(2026, 11, 1)) == 0       # returned on the due date
    assert circulation.fine_for(due, date(2026, 10, 20)) == 0      # early
    assert circulation.fine_for(due, date(2026, 11, 4)) == 3 * FINE_PER_DAY


def test_issue_sets_due_date_and_reduces_availability(fresh_conn):
    sid, bid = _free_student(fresh_conn), _free_book(fresh_conn)
    before = db.copies_available(fresh_conn, bid)
    out = circulation.issue_book(fresh_conn, sid, bid)
    assert out.ok and out.loan_id
    assert db.copies_available(fresh_conn, bid) == before - 1
    due = fresh_conn.execute("SELECT due_on FROM loans WHERE loan_id = ?", (out.loan_id,)).fetchone()["due_on"]
    assert due == (AS_OF + timedelta(days=LOAN_DAYS)).isoformat()


def test_cannot_issue_when_no_copy_is_free(fresh_conn):
    sid = _free_student(fresh_conn)
    full = db.availability_df(fresh_conn).query("available == 0")
    assert not full.empty, "the simulated data should contain at least one fully borrowed book"
    out = circulation.issue_book(fresh_conn, sid, int(full.iloc[0]["book_id"]))
    assert not out.ok and "available" in out.message


def test_cannot_exceed_the_borrowing_limit(fresh_conn):
    sid = _free_student(fresh_conn)
    free_books = db.availability_df(fresh_conn).query("available > 0")["book_id"].tolist()
    for bid in free_books[:MAX_ACTIVE_LOANS]:
        assert circulation.issue_book(fresh_conn, sid, int(bid)).ok
    out = circulation.issue_book(fresh_conn, sid, int(free_books[MAX_ACTIVE_LOANS]))
    assert not out.ok and "limit" in out.message


def test_cannot_borrow_same_book_twice(fresh_conn):
    sid, bid = _free_student(fresh_conn), _free_book(fresh_conn)
    assert circulation.issue_book(fresh_conn, sid, bid).ok
    assert not circulation.issue_book(fresh_conn, sid, bid).ok


def test_student_with_overdue_book_is_blocked(fresh_conn):
    sid = int(circulation.overdue_loans(fresh_conn).iloc[0]["student_id"])
    out = circulation.issue_book(fresh_conn, sid, _free_book(fresh_conn))
    assert not out.ok and "overdue" in out.message


def test_return_frees_the_copy_and_charges_fine(fresh_conn):
    late = circulation.overdue_loans(fresh_conn).iloc[0]
    before = db.copies_available(fresh_conn, int(late["book_id"]))
    out = circulation.return_book(fresh_conn, int(late["loan_id"]))
    assert out.ok and out.fine == int(late["days_overdue"]) * FINE_PER_DAY > 0
    assert db.copies_available(fresh_conn, int(late["book_id"])) == before + 1
    assert not circulation.return_book(fresh_conn, int(late["loan_id"])).ok  # can't return twice


def test_renewal_rules(fresh_conn):
    sid, bid = _free_student(fresh_conn), _free_book(fresh_conn)
    loan_id = circulation.issue_book(fresh_conn, sid, bid).loan_id
    assert circulation.renew_loan(fresh_conn, loan_id).ok
    assert not circulation.renew_loan(fresh_conn, loan_id).ok            # only one renewal allowed
    late_id = int(circulation.overdue_loans(fresh_conn).iloc[0]["loan_id"])
    assert not circulation.renew_loan(fresh_conn, late_id).ok            # overdue loans can't be renewed


def test_overdue_list_is_consistent(conn):
    overdue = circulation.overdue_loans(conn)
    assert len(overdue) > 0
    assert (overdue["days_overdue"] > 0).all()
    assert (overdue["fine"] == overdue["days_overdue"] * FINE_PER_DAY).all()
    assert overdue["days_overdue"].is_monotonic_decreasing
    # must agree with the simpler query that the chatbot and app already use
    assert set(overdue["loan_id"]) == set(db.overdue_loans_df(conn, as_of=AS_OF)["loan_id"])


def test_reminders_are_created_once_per_day(fresh_conn):
    first = circulation.create_reminders(fresh_conn)
    assert first["overdue"] > 0
    second = circulation.create_reminders(fresh_conn)           # same day again
    assert second == {"due_soon": 0, "overdue": 0}
    assert len(db.notifications_df(fresh_conn)) == first["due_soon"] + first["overdue"]


def test_library_stats(conn):
    stats = circulation.library_stats(conn)
    assert stats["titles"] == 100 and stats["students"] == 150
    assert 0 < stats["utilisation"] < 1
    assert stats["copies_out"] == len(circulation.open_loans(conn))
    assert stats["overdue"] == len(circulation.overdue_loans(conn))