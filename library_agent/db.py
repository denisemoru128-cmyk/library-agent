"""Database layer (SQLite).

This module is the ONLY place that knows about SQL table layouts. The AI
modules ask it for pandas DataFrames and never write SQL themselves, which keeps
the "intelligence" code easy to read and easy to test.

Security note: every query that includes a value uses ``?`` placeholders and
never builds SQL with f-strings, so user input can not inject SQL.
"""
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .config import DB_PATH, LOAN_DAYS, MAX_ACTIVE_LOANS

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    author      TEXT NOT NULL,
    genre       TEXT NOT NULL,
    description TEXT NOT NULL,
    copies      INTEGER NOT NULL CHECK (copies >= 0)
);

CREATE TABLE IF NOT EXISTS students (
    student_id  INTEGER PRIMARY KEY,
    reg_no      TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    department  TEXT NOT NULL,
    interests   TEXT NOT NULL DEFAULT ''      -- genres declared at registration, joined by '|'
);

CREATE TABLE IF NOT EXISTS loans (
    loan_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER NOT NULL REFERENCES students(student_id),
    book_id     INTEGER NOT NULL REFERENCES books(book_id),
    borrowed_on TEXT NOT NULL,                -- ISO dates sort correctly as text
    due_on      TEXT NOT NULL,
    returned_on TEXT,                         -- NULL means the book is still out
    renewals    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_loans_student ON loans(student_id);
CREATE INDEX IF NOT EXISTS idx_loans_book    ON loans(book_id);
CREATE INDEX IF NOT EXISTS idx_loans_open    ON loans(returned_on);

CREATE TABLE IF NOT EXISTS notifications (
    notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER NOT NULL REFERENCES students(student_id),
    loan_id     INTEGER REFERENCES loans(loan_id),
    kind        TEXT NOT NULL,                -- 'due_soon' or 'overdue'
    message     TEXT NOT NULL,
    created_on  TEXT NOT NULL,
    -- one message of each kind per loan per day: re-running the agent is harmless
    UNIQUE (loan_id, kind, created_on)
);

CREATE TABLE IF NOT EXISTS agent_log (
    log_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_on  TEXT NOT NULL,
    action  TEXT NOT NULL,
    detail  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(path=None) -> sqlite3.Connection:
    """Open a connection. A fresh one per use is cheap in SQLite and avoids
    thread problems with Streamlit, which reruns scripts on different threads."""
    path = Path(path) if path else DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


# ------------------------------------------------------------------ the clock
def today(conn: sqlite3.Connection) -> date:
    """The library's current date. It is stored in the database so a demo stays
    consistent (the simulated history always ends 'today')."""
    row = conn.execute("SELECT value FROM meta WHERE key = 'as_of'").fetchone()
    return date.fromisoformat(row["value"]) if row else date.today()


def set_today(conn: sqlite3.Connection, d: date) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('as_of', ?)", (d.isoformat(),))
    conn.commit()


# ---------------------------------------------------------------- DataFrames
def books_df(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM books ORDER BY book_id", conn)


def students_df(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM students ORDER BY student_id", conn)


def loans_df(conn) -> pd.DataFrame:
    """All loans, with the three date columns converted to real datetimes."""
    df = pd.read_sql_query("SELECT * FROM loans ORDER BY loan_id", conn)
    for col in ("borrowed_on", "due_on", "returned_on"):
        df[col] = pd.to_datetime(df[col])
    return df


def availability_df(conn) -> pd.DataFrame:
    """One row per book: total copies, copies currently out, copies available."""
    return pd.read_sql_query(
        """
        SELECT b.book_id, b.copies,
               COUNT(l.loan_id)                   AS copies_out,
               b.copies - COUNT(l.loan_id)        AS available
        FROM books b
        LEFT JOIN loans l ON l.book_id = b.book_id AND l.returned_on IS NULL
        GROUP BY b.book_id
        ORDER BY b.book_id
        """,
        conn,
    )


def overdue_loans_df(conn, as_of: date | None = None, student_id: int | None = None) -> pd.DataFrame:
    """Open loans whose due date has passed, with the number of overdue days."""
    as_of = as_of or today(conn)
    return pd.read_sql_query(
        """
        SELECT l.loan_id, l.student_id, s.name AS student_name, s.reg_no,
               l.book_id, b.title, b.author, l.borrowed_on, l.due_on,
               CAST(julianday(?) - julianday(l.due_on) AS INTEGER) AS days_overdue
        FROM loans l
        JOIN books b ON b.book_id = l.book_id
        JOIN students s ON s.student_id = l.student_id
        WHERE l.returned_on IS NULL AND l.due_on < ?
          AND (? IS NULL OR l.student_id = ?)
        ORDER BY l.due_on, l.loan_id
        """,
        conn,
        params=(as_of.isoformat(), as_of.isoformat(), student_id, student_id),
        parse_dates=["borrowed_on", "due_on"],
    )


def due_loans_df(conn, start: date, end: date, student_id: int | None = None) -> pd.DataFrame:
    """Open loans due within an inclusive date range."""
    if end < start:
        raise ValueError("end must be on or after start")
    return pd.read_sql_query(
        """
        SELECT l.loan_id, l.student_id, s.name AS student_name, s.reg_no,
               l.book_id, b.title, b.author, l.borrowed_on, l.due_on
        FROM loans l
        JOIN books b ON b.book_id = l.book_id
        JOIN students s ON s.student_id = l.student_id
        WHERE l.returned_on IS NULL AND l.due_on >= ? AND l.due_on <= ?
          AND (? IS NULL OR l.student_id = ?)
        ORDER BY l.due_on, l.loan_id
        """,
        conn,
        params=(start.isoformat(), end.isoformat(), student_id, student_id),
        parse_dates=["borrowed_on", "due_on"],
    )


def student_loans_df(conn, student_id: int, open_only: bool = True) -> pd.DataFrame:
    """Return one student's loans, optionally limited to books not yet returned."""
    return pd.read_sql_query(
        """
        SELECT l.loan_id, l.student_id, l.book_id, b.title, b.author,
               l.borrowed_on, l.due_on, l.returned_on, l.renewals
        FROM loans l
        JOIN books b ON b.book_id = l.book_id
        WHERE l.student_id = ? AND (? = 0 OR l.returned_on IS NULL)
        ORDER BY l.due_on, l.loan_id
        """,
        conn,
        params=(student_id, int(open_only)),
        parse_dates=["borrowed_on", "due_on", "returned_on"],
    )


# ------------------------------------------------------------- single records
def get_book(conn, book_id: int):
    return conn.execute("SELECT * FROM books WHERE book_id = ?", (book_id,)).fetchone()


def get_student(conn, student_id: int):
    return conn.execute("SELECT * FROM students WHERE student_id = ?", (student_id,)).fetchone()


def add_book(conn, title: str, author: str, genre: str, description: str, copies: int) -> int:
    """Add a book to the catalogue and return its new id."""
    cur = conn.execute(
        "INSERT INTO books (title, author, genre, description, copies) VALUES (?, ?, ?, ?, ?)",
        (title.strip(), author.strip(), genre.strip(), description.strip(), int(copies)),
    )
    conn.commit()
    return cur.lastrowid


def borrow_book(conn, student_id: int, book_id: int, borrowed_on: date | None = None) -> int:
    """Create a loan if the student and a copy are available; return its id."""
    borrowed_on = borrowed_on or today(conn)
    due_on = borrowed_on + timedelta(days=LOAN_DAYS)
    with conn:
        if get_student(conn, student_id) is None:
            raise ValueError(f"Unknown student id: {student_id}")
        if get_book(conn, book_id) is None:
            raise ValueError(f"Unknown book id: {book_id}")

        active = conn.execute(
            "SELECT COUNT(*) FROM loans WHERE student_id = ? AND returned_on IS NULL",
            (student_id,),
        ).fetchone()[0]
        if active >= MAX_ACTIVE_LOANS:
            raise ValueError(f"A student may hold at most {MAX_ACTIVE_LOANS} active loans")

        available = conn.execute(
            """
            SELECT b.copies - COUNT(l.loan_id)
            FROM books b
            LEFT JOIN loans l ON l.book_id = b.book_id AND l.returned_on IS NULL
            WHERE b.book_id = ?
            GROUP BY b.book_id
            """,
            (book_id,),
        ).fetchone()[0]
        if available <= 0:
            raise ValueError("No copies of this book are currently available")

        cur = conn.execute(
            """
            INSERT INTO loans (student_id, book_id, borrowed_on, due_on)
            VALUES (?, ?, ?, ?)
            """,
            (student_id, book_id, borrowed_on.isoformat(), due_on.isoformat()),
        )
    return cur.lastrowid


def record_return(conn, loan_id: int, returned_on: date | None = None) -> None:
    """Mark an active loan returned; fail explicitly for unknown/already returned loans."""
    returned_on = returned_on or today(conn)
    with conn:
        loan = conn.execute(
            "SELECT borrowed_on, returned_on FROM loans WHERE loan_id = ?",
            (loan_id,),
        ).fetchone()
        if loan is None:
            raise ValueError(f"Unknown loan id: {loan_id}")
        if loan["returned_on"] is not None:
            raise ValueError(f"Loan {loan_id} has already been returned")
        if returned_on < date.fromisoformat(loan["borrowed_on"]):
            raise ValueError("returned_on must be on or after borrowed_on")
        conn.execute(
            "UPDATE loans SET returned_on = ? WHERE loan_id = ? AND returned_on IS NULL",
            (returned_on.isoformat(), loan_id),
        )


# ---------------------------------------------------- helpers for circulation.py
# circulation.py holds the library RULES; everything that touches SQL stays here.
def open_loans_df(conn, student_id: int | None = None) -> pd.DataFrame:
    """Every loan that is still out (optionally for one student), earliest due date first."""
    return pd.read_sql_query(
        """
        SELECT l.loan_id, l.student_id, s.name AS student_name, s.reg_no,
               l.book_id, b.title, b.author, l.borrowed_on, l.due_on, l.renewals
        FROM loans l
        JOIN books b ON b.book_id = l.book_id
        JOIN students s ON s.student_id = l.student_id
        WHERE l.returned_on IS NULL AND (? IS NULL OR l.student_id = ?)
        ORDER BY l.due_on, l.loan_id
        """,
        conn,
        params=(student_id, student_id),
        parse_dates=["borrowed_on", "due_on"],
    )


def get_loan(conn, loan_id: int):
    """One loan joined with its book title, or None."""
    return conn.execute(
        "SELECT l.*, b.title FROM loans l JOIN books b ON b.book_id = l.book_id WHERE l.loan_id = ?",
        (loan_id,),
    ).fetchone()


def copies_available(conn, book_id: int) -> int:
    row = conn.execute(
        "SELECT b.copies - COUNT(l.loan_id) AS free FROM books b "
        "LEFT JOIN loans l ON l.book_id = b.book_id AND l.returned_on IS NULL "
        "WHERE b.book_id = ? GROUP BY b.book_id",
        (book_id,),
    ).fetchone()
    return row["free"] if row else 0


def extend_loan(conn, loan_id: int, new_due_on: date) -> None:
    """Push the due date back and count one renewal."""
    conn.execute(
        "UPDATE loans SET due_on = ?, renewals = renewals + 1 WHERE loan_id = ?",
        (new_due_on.isoformat(), loan_id),
    )


def count_loans_since(conn, since: date) -> int:
    return conn.execute("SELECT COUNT(*) FROM loans WHERE borrowed_on > ?", (since.isoformat(),)).fetchone()[0]


def insert_notification(conn, student_id: int, loan_id: int, kind: str, message: str, created_on: date) -> bool:
    """Store a message for a student. The UNIQUE(loan_id, kind, created_on) rule in the
    schema makes this a no-op if the same message was already sent today; returns
    True only when a NEW message was stored."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO notifications (student_id, loan_id, kind, message, created_on) "
        "VALUES (?, ?, ?, ?, ?)",
        (int(student_id), int(loan_id), kind, message, created_on.isoformat()),
    )
    return cur.rowcount == 1


def notifications_df(conn, student_id: int | None = None, limit: int = 200) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT n.notification_id, n.created_on, s.name AS student_name, n.kind, n.message
        FROM notifications n JOIN students s ON s.student_id = n.student_id
        WHERE (? IS NULL OR n.student_id = ?)
        ORDER BY n.notification_id DESC LIMIT ?
        """,
        conn,
        params=(student_id, student_id, limit),
    )