"""Simulated library data.

The assignment asks for simulated data, but random numbers would give the AI
nothing to learn. So the simulator builds in the patterns a real library has:

* every student has favourite genres plus department genres (taste),
* within a genre each student likes different books (so collaborative
  filtering has something to find),
* a few books are much more popular than others (long tail),
* borrowing rises before exams, drops in breaks, and slowly grows over time
  (so the demand forecaster has seasonality and a trend to learn),
* some students return late (so there are real overdue loans and fines).

Same random seed -> exactly the same dataset, so results are repeatable.
"""
import csv
from collections import defaultdict
from datetime import date, timedelta

import numpy as np

from . import db
from .academic_calendar import phase
from .config import (
    BOOKS_CSV, DB_PATH, HISTORY_DAYS, LOAN_DAYS, MAX_ACTIVE_LOANS, N_STUDENTS, RANDOM_SEED,
)

ACADEMIC_GENRES = {
    "Computer Science", "Mathematics & Statistics", "Engineering", "Business & Economics",
    "Health & Medicine", "Law & Governance", "Natural Sciences",
}
LEISURE_GENRES = ["African Literature", "World Fiction", "History & Politics", "Self-Development"]

# department -> the academic genres its students mostly study
DEPARTMENTS = {
    "Computer Science": ["Computer Science", "Mathematics & Statistics"],
    "Engineering": ["Engineering", "Mathematics & Statistics"],
    "Business": ["Business & Economics"],
    "Medicine & Health": ["Health & Medicine", "Natural Sciences"],
    "Law": ["Law & Governance", "History & Politics"],
    "Arts & Humanities": ["African Literature", "World Fiction", "History & Politics"],
    "Environmental Science": ["Natural Sciences", "Mathematics & Statistics"],
}

FIRST_NAMES = (
    "Brian Faith Kevin Mercy Dennis Grace Collins Joy Eric Lucy Victor Sharon Felix Esther Samuel Naomi "
    "Allan Ruth Peter Cynthia James Mary Joseph Janet Daniel Purity Martin Winnie Stephen Caroline Isaac "
    "Beatrice Moses Irene Jeremiah Lilian Amos Rose Patrick Diana Edwin Judy Duncan Millicent Kelvin Nancy "
    "Festus Carol Wilson Agnes Hassan Amina Abdi Fatuma Ali Halima"
).split()
LAST_NAMES = (
    "Mwangi Otieno Wanjiku Kamau Odhiambo Njeri Kiprop Chebet Mutua Wafula Achieng Kariuki Omondi Nyambura "
    "Kipchoge Muthoni Onyango Wekesa Naliaka Maina Ochieng Akinyi Korir Jepkoech Mbugua Wambui Kimani Atieno "
    "Barasa Cheruiyot Githinji Mohamed Hussein Ndungu Okoth Rotich Waweru Simiyu"
).split()


def load_catalog(path=BOOKS_CSV) -> list[dict]:
    """Read the sample catalogue (data/books.csv)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["book_id"] = int(r["book_id"])
        r["copies"] = int(r["copies"])
    return rows


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def make_students(rng, n: int) -> list[dict]:
    """Create n students, each with a department and 1-2 favourite genres."""
    students, used = [], set()
    while len(students) < n:
        name = f"{_pick(rng, FIRST_NAMES)} {_pick(rng, LAST_NAMES)}"
        if name in used:
            continue
        used.add(name)
        dept = _pick(rng, list(DEPARTMENTS))
        favourites = [_pick(rng, DEPARTMENTS[dept])]
        if rng.random() < 0.7:  # most students also have a leisure-reading interest
            extra = _pick(rng, LEISURE_GENRES)
            if extra not in favourites:
                favourites.append(extra)
        students.append({
            "student_id": len(students) + 1,
            "reg_no": f"REG-{len(students) + 1:04d}",
            "name": name,
            "department": dept,
            "favourites": favourites,
            "interests": "|".join(favourites),
        })
    return students


def seasonal_factor(d: date, genre: str) -> float:
    """How strongly the academic calendar pushes borrowing of this genre on day d."""
    p, academic = phase(d), genre in ACADEMIC_GENRES
    if p.is_break:
        return 0.25 if academic else 0.55     # leisure reading holds up better in breaks
    if p.is_exam:
        return 1.9 if academic else 0.7       # everyone borrows textbooks before exams
    if p.week_of_term <= 2:
        return 1.2 if academic else 1.0       # start-of-term rush
    return 1.0


def simulate_loans(rng, books, students, start: date, end: date) -> list[dict]:
    """Walk through the calendar day by day and let each student borrow."""
    n_books, n_students = len(books), len(students)
    genres = [b["genre"] for b in books]
    copies = np.array([b["copies"] for b in books])

    # Hidden "taste" model: global popularity x personal taste x genre preference.
    popularity = rng.lognormal(0.0, 0.8, n_books)
    taste = rng.lognormal(0.0, 0.9, (n_students, n_books))
    genre_boost = np.ones((n_students, n_books))
    for i, s in enumerate(students):
        for j, g in enumerate(genres):
            if g in s["favourites"]:
                genre_boost[i, j] = 12.0
            elif g in DEPARTMENTS[s["department"]]:
                genre_boost[i, j] = 4.0
    weights = popularity * taste * genre_boost
    activity = rng.lognormal(0.0, 0.4, n_students)  # some students read more than others

    copies_out = np.zeros(n_books, dtype=int)
    active = np.zeros(n_students, dtype=int)
    borrowed_before = np.zeros((n_students, n_books), dtype=bool)
    holding = np.zeros((n_students, n_books), dtype=bool)  # a student never holds two copies of one book
    returns = defaultdict(list)  # date -> [(student_idx, book_idx)] coming back that day
    loans = []

    total_days = (end - start).days
    for offset in range(total_days + 1):
        d = start + timedelta(days=offset)
        for s_idx, b_idx in returns.pop(d, []):
            copies_out[b_idx] -= 1
            active[s_idx] -= 1
            holding[s_idx, b_idx] = False
        trend = 1.0 + 0.25 * offset / total_days  # library use grows ~25% over the period

        for s_idx in range(n_students):
            for _ in range(rng.poisson(0.17 * activity[s_idx])):
                if active[s_idx] >= MAX_ACTIVE_LOANS:
                    break
                w = (weights[s_idx] * np.where(borrowed_before[s_idx], 0.1, 1.0)
                     * (copies_out < copies) * ~holding[s_idx])
                if w.sum() == 0:
                    continue
                b_idx = int(rng.choice(n_books, p=w / w.sum()))
                if rng.random() > min(1.0, seasonal_factor(d, genres[b_idx]) / 2.4 * trend):
                    continue

                due = d + timedelta(days=LOAN_DAYS)
                renewals = 1 if rng.random() < 0.08 else 0
                if renewals:
                    due += timedelta(days=LOAN_DAYS)
                r = rng.random()
                if r < 0.80:                       # on time
                    back = d + timedelta(days=int(rng.integers(3, LOAN_DAYS + 1)))
                elif r < 0.95:                     # a little late
                    back = due + timedelta(days=int(rng.integers(1, 11)))
                else:                              # very late
                    back = due + timedelta(days=int(rng.integers(11, 36)))

                copies_out[b_idx] += 1
                active[s_idx] += 1
                borrowed_before[s_idx, b_idx] = True
                holding[s_idx, b_idx] = True
                returned = back if back <= end else None
                if returned:
                    returns[back].append((s_idx, b_idx))
                loans.append({
                    "student_id": students[s_idx]["student_id"],
                    "book_id": books[b_idx]["book_id"],
                    "borrowed_on": d.isoformat(),
                    "due_on": due.isoformat(),
                    "returned_on": returned.isoformat() if returned else None,
                    "renewals": renewals,
                })
    return loans


def seed_database(db_path=DB_PATH, as_of: date | None = None, seed: int = RANDOM_SEED,
                  n_students: int = N_STUDENTS, history_days: int = HISTORY_DAYS) -> dict:
    """Build a brand-new database file full of simulated data. Returns a summary."""
    as_of = as_of or date.today()
    rng = np.random.default_rng(seed)
    books = load_catalog()
    students = make_students(rng, n_students)
    loans = simulate_loans(rng, books, students, as_of - timedelta(days=history_days), as_of)

    if db_path.exists():
        db_path.unlink()
    conn = db.connect(db_path)
    db.init_schema(conn)
    conn.executemany(
        "INSERT INTO books (book_id, title, author, genre, description, copies) VALUES (?, ?, ?, ?, ?, ?)",
        [(b["book_id"], b["title"], b["author"], b["genre"], b["description"], b["copies"]) for b in books],
    )
    conn.executemany(
        "INSERT INTO students (student_id, reg_no, name, department, interests) VALUES (?, ?, ?, ?, ?)",
        [(s["student_id"], s["reg_no"], s["name"], s["department"], s["interests"]) for s in students],
    )
    conn.executemany(
        "INSERT INTO loans (student_id, book_id, borrowed_on, due_on, returned_on, renewals) "
        "VALUES (:student_id, :book_id, :borrowed_on, :due_on, :returned_on, :renewals)",
        loans,
    )
    db.set_today(conn, as_of)
    summary = {
        "as_of": as_of.isoformat(),
        "books": len(books),
        "students": len(students),
        "loans": len(loans),
        "currently_out": sum(1 for l in loans if l["returned_on"] is None),
        "overdue_now": sum(1 for l in loans if l["returned_on"] is None and l["due_on"] < as_of.isoformat()),
    }
    conn.close()
    return summary