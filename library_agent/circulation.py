"""Circulation rules: issuing, returning and renewing books, overdue tracking,
fines and reminders.

This is deliberately ordinary, rule-based logic (no machine learning): library
policy is a set of exact rules, and rules are the right tool for exact rules.
The agent (agent.py), the chatbot and the interface all call these functions
instead of re-implementing the rules, so the policy lives in exactly one place.

Division of work:
    config.py       the numbers (loan days, fine per day, limits)
    db.py           all SQL
    circulation.py  the rules: "may this student borrow this book?"
Every change happens inside one transaction, so two people can not both take
the last copy.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from . import db
from .config import FINE_PER_DAY, LOAN_DAYS, MAX_ACTIVE_LOANS, MAX_RENEWALS, REMINDER_DAYS_BEFORE_DUE


@dataclass
class Outcome:
    """The result of an action, in a form the UI, the chatbot and the agent can all show."""
    ok: bool
    message: str
    loan_id: int | None = None
    fine: int = 0


@contextmanager
def _transaction(conn):
    """BEGIN IMMEDIATE locks the database for writing before we check anything."""
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def fine_for(due_on: date, until: date) -> int:
    """KES owed for a book due on ``due_on`` and returned (or still out) on ``until``."""
    return max((until - due_on).days, 0) * FINE_PER_DAY


# --------------------------------------------------------------- queries
def open_loans(conn, student_id: int | None = None, on: date | None = None) -> pd.DataFrame:
    """Loans still out, with days_left (negative = overdue), the fine so far and a status."""
    on = on or db.today(conn)
    df = db.open_loans_df(conn, student_id)
    due = pd.to_datetime(df["due_on"]).dt.date
    df["days_left"] = [(d - on).days for d in due]
    df["fine"] = [fine_for(d, on) for d in due]
    df["status"] = ["overdue" if x < 0 else ("due soon" if x <= REMINDER_DAYS_BEFORE_DUE else "on loan")
                    for x in df["days_left"]]
    return df


def overdue_loans(conn, on: date | None = None) -> pd.DataFrame:
    """Every loan past its due date, worst first."""
    df = open_loans(conn, on=on)
    df = df[df["days_left"] < 0].copy()
    df["days_overdue"] = -df["days_left"]
    return df.sort_values("days_overdue", ascending=False).reset_index(drop=True)


def due_soon(conn, on: date | None = None, within_days: int = REMINDER_DAYS_BEFORE_DUE) -> pd.DataFrame:
    """Loans that fall due today or within the next ``within_days`` days."""
    df = open_loans(conn, on=on)
    return df[(df["days_left"] >= 0) & (df["days_left"] <= within_days)].reset_index(drop=True)


def student_summary(conn, student_id: int, on: date | None = None) -> dict:
    """A student's position in one place, so every screen and the chatbot say the same thing.

    ``slots_free`` is how many more books the limit allows. ``can_borrow_more`` is what the student
    can really borrow now: zero while any book is overdue, because the rules pause borrowing.
    ``blocking_title`` is the book to return first (the longest overdue), or None.
    """
    loans = open_loans(conn, student_id, on)
    late = loans[loans["days_left"] < 0].sort_values("days_left")
    slots_free = max(MAX_ACTIVE_LOANS - len(loans), 0)
    paused = not late.empty
    return {
        "active_loans": len(loans),
        "slots_free": slots_free,
        "can_borrow_more": 0 if paused else slots_free,
        "borrowing_paused": paused,
        "blocking_title": None if late.empty else str(late.iloc[0]["title"]),
        "overdue": len(late),
        "fines_owed": int(loans["fine"].sum()),
    }


def return_outlook(conn, book_id: int, on: date | None = None, loans: pd.DataFrame | None = None) -> dict:
    """When will a book that is fully out come back?

    ``next_due`` is the earliest due date among copies that are still on time (None if there is
    none). ``late_copies`` are copies already past their due date: they have no reliable return date.
    Pass ``loans`` (from open_loans) when asking about many books, so they are loaded only once.
    """
    loans = open_loans(conn, on=on) if loans is None else loans
    loans = loans[loans["book_id"] == book_id]
    on_time = loans[loans["days_left"] >= 0].sort_values("days_left")
    return {
        "copies_out": len(loans),
        "next_due": None if on_time.empty else on_time.iloc[0]["due_on"].date(),
        "late_copies": int((loans["days_left"] < 0).sum()),
    }


# ------------------------------------------------- why an action is not allowed
# One function decides, two callers use the answer: the action itself (a full sentence in its Outcome)
# and the screens (a short hint under a disabled button). They can not disagree, because both ask here.
BORROW_HINTS = {
    "no_copy": "No free copy right now",
    "limit": f"You have {MAX_ACTIVE_LOANS} books out, the limit",
    "overdue": "Return your overdue book first",
    "has_it": "Already on your loans",
}
RENEW_HINTS = {
    "overdue": "Overdue: return it instead",
    "other_overdue": "Return the overdue book first",
    "max_renewals": "Already renewed once" if MAX_RENEWALS == 1 else f"Renewed {MAX_RENEWALS} times already",
}


def _borrow_problem(loans: pd.DataFrame, book_id: int, copies_free: int) -> str | None:
    """The first borrowing rule that fails (a key of BORROW_HINTS), or None. ``loans`` are the student's."""
    if copies_free < 1:
        return "no_copy"
    if len(loans) >= MAX_ACTIVE_LOANS:
        return "limit"
    if (loans["days_left"] < 0).any():
        return "overdue"
    if (loans["book_id"] == book_id).any():
        return "has_it"
    return None


def _renew_problem(overdue: bool, other_overdue: bool, renewals: int) -> str | None:
    """The first renewing rule that fails (a key of RENEW_HINTS), or None."""
    if overdue:
        return "overdue"
    if other_overdue:
        return "other_overdue"
    if renewals >= MAX_RENEWALS:
        return "max_renewals"
    return None


def borrow_blocker(loans: pd.DataFrame, book_id: int, copies_free: int) -> str | None:
    """Short reason this student can not borrow this book now (for a button hint), or None.
    ``loans`` is the student's open_loans frame."""
    problem = _borrow_problem(loans, book_id, copies_free)
    return None if problem is None else BORROW_HINTS[problem]


def renew_blocker(loans: pd.DataFrame, loan_id: int) -> str | None:
    """Short reason this loan can not be renewed now, or None. ``loans`` is the student's open_loans frame."""
    row = loans[loans["loan_id"] == loan_id].iloc[0]
    other_late = loans[(loans["loan_id"] != loan_id) & (loans["days_left"] < 0)]
    problem = _renew_problem(row["days_left"] < 0, not other_late.empty, int(row["renewals"]))
    return None if problem is None else RENEW_HINTS[problem]


# --------------------------------------------------------------- actions
def issue_book(conn, student_id: int, book_id: int, on: date | None = None) -> Outcome:
    """Lend a book if every library rule allows it; otherwise say which rule failed."""
    on = on or db.today(conn)
    with _transaction(conn):
        student, book = db.get_student(conn, student_id), db.get_book(conn, book_id)
        if student is None:
            return Outcome(False, "Student not found.")
        if book is None:
            return Outcome(False, "Book not found.")
        loans = open_loans(conn, student_id, on)
        problem = _borrow_problem(loans, book_id, db.copies_available(conn, book_id))
        if problem:
            return Outcome(False, {
                "no_copy": f"No copy of “{book['title']}” is available right now.",
                "limit": f"{student['name']} already has {MAX_ACTIVE_LOANS} books out (the limit).",
                "overdue": f"{student['name']} has overdue books and must return them first.",
                "has_it": f"{student['name']} already has “{book['title']}” on loan.",
            }[problem])
        try:
            loan_id = db.borrow_book(conn, student_id, book_id, borrowed_on=on)
        except ValueError as error:  # a rule inside db.borrow_book said no
            return Outcome(False, str(error))
        due = on + timedelta(days=LOAN_DAYS)
        return Outcome(True, f"Issued “{book['title']}” to {student['name']}. Due {due.isoformat()}.",
                       loan_id=loan_id)


def return_book(conn, loan_id: int, on: date | None = None) -> Outcome:
    """Record a return and work out any fine."""
    on = on or db.today(conn)
    with _transaction(conn):
        loan = db.get_loan(conn, loan_id)
        if loan is None:
            return Outcome(False, "Loan not found.")
        if loan["returned_on"] is not None:
            return Outcome(False, "This book was already returned.")
        try:
            db.record_return(conn, loan_id, returned_on=on)
        except ValueError as error:
            return Outcome(False, str(error))
        fine = fine_for(date.fromisoformat(loan["due_on"]), on)
        note = f" Fine: KES {fine}." if fine else " No fine."
        return Outcome(True, f"Returned “{loan['title']}”.{note}", loan_id=loan_id, fine=fine)


def renew_loan(conn, loan_id: int, on: date | None = None) -> Outcome:
    """Extend a loan by one loan period, if the rules allow."""
    on = on or db.today(conn)
    with _transaction(conn):
        loan = db.get_loan(conn, loan_id)
        if loan is None or loan["returned_on"] is not None:
            return Outcome(False, "That loan is not active.")
        # The rule the app shows to students: any overdue book pauses borrowing AND renewing.
        others = open_loans(conn, loan["student_id"], on)
        late = others[others["days_left"] < 0].sort_values("days_left")
        problem = _renew_problem(date.fromisoformat(loan["due_on"]) < on, not late.empty, loan["renewals"])
        if problem:
            return Outcome(False, {
                "overdue": "Overdue books can not be renewed. Please return it and pay the fine.",
                "other_overdue": f"“{loan['title']}” can not be renewed while "
                                 f"“{late.iloc[0]['title'] if not late.empty else ''}” is overdue. "
                                 "Return that book first.",
                "max_renewals": f"“{loan['title']}” has already been renewed {MAX_RENEWALS} time(s).",
            }[problem])
        new_due = on + timedelta(days=LOAN_DAYS)
        db.extend_loan(conn, loan_id, new_due)
        return Outcome(True, f"Renewed “{loan['title']}”. New due date: {new_due.isoformat()}.",
                       loan_id=loan_id)


# ------------------------------------------------------------ notifications
def create_reminders(conn, on: date | None = None) -> dict:
    """Write a message for every loan that is due soon or overdue.

    The database allows one message of each kind per loan per day, so running this
    twice on the same day creates nothing new. Returns how many NEW messages were made.
    """
    on = on or db.today(conn)
    created = {"due_soon": 0, "overdue": 0}
    for row in due_soon(conn, on).itertuples():
        when = "today" if row.days_left == 0 else f"in {row.days_left} day(s)"
        created["due_soon"] += db.insert_notification(
            conn, row.student_id, row.loan_id, "due_soon",
            f"Reminder: “{row.title}” is due {when} ({row.due_on.date()}).", on)
    for row in overdue_loans(conn, on).itertuples():
        created["overdue"] += db.insert_notification(
            conn, row.student_id, row.loan_id, "overdue",
            f"Overdue: “{row.title}” was due on {row.due_on.date()} ({row.days_overdue} days ago). "
            f"Fine so far: KES {row.fine}. Please return it.", on)
    conn.commit()
    return created


# ------------------------------------------------------------ librarian view
def library_stats(conn, on: date | None = None) -> dict:
    """Headline numbers for the librarian dashboard."""
    on = on or db.today(conn)
    loans = open_loans(conn, on=on)
    total_copies = int(db.books_df(conn)["copies"].sum())
    return {
        "titles": len(db.books_df(conn)),
        "copies": total_copies,
        "copies_out": len(loans),
        "utilisation": len(loans) / total_copies if total_copies else 0.0,
        "overdue": int((loans["days_left"] < 0).sum()),
        "fines_outstanding": int(loans["fine"].sum()),
        "loans_last_7_days": db.count_loans_since(conn, on - timedelta(days=7)),
        "students": len(db.students_df(conn)),
    }