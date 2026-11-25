"""Small HTML building blocks the screens share: the stat strip, a book line with its generated cover,
and the wording for due dates and availability.

Everything here only *describes* a situation (it never changes the library). The decisions come from
``library_agent.circulation``, so the screens, the chatbot and the rules say the same thing.
"""
import html

import pandas as pd
import streamlit as st

from library_agent import circulation, dates
from library_agent.config import FINE_PER_DAY, MAX_ACTIVE_LOANS

from . import theme


def esc(text) -> str:
    return html.escape(str(text))


def plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


# ------------------------------------------------------------------ stat strip
def stat_strip(tiles: list[dict]) -> None:
    """A row of headline numbers. Each tile: label, value, optional unit, sub-line and tone ('late' = red)."""
    cells = ""
    for tile in tiles:
        unit = f' <i>{esc(tile["unit"])}</i>' if tile.get("unit") else ""
        cells += (f'<div class="cell {tile.get("tone", "")}"><small>{esc(tile["label"])}</small>'
                  f'<b>{esc(tile["value"])}{unit}</b><span>{esc(tile.get("sub", ""))}</span></div>')
    st.html(f'<div class="strip">{cells}</div>')


def student_tiles(summary: dict, loans: pd.DataFrame) -> list[dict]:
    """The four tiles at the top of Home and My loans, from circulation.student_summary."""
    late = loans[loans["days_left"] < 0]
    on_time = loans[loans["days_left"] >= 0].sort_values("days_left")
    worst = int(-late["days_left"].min()) if not late.empty else 0

    on_loan = {"label": "On loan", "value": summary["active_loans"], "unit": f"of {MAX_ACTIVE_LOANS}"}
    if summary["overdue"]:
        on_loan["sub"] = f"{summary['overdue']} of them overdue"
    else:
        on_loan["sub"] = ("no slots free" if summary["slots_free"] == 0
                          else f"{plural(summary['slots_free'], 'slot')} free")

    overdue = ({"label": "Overdue", "value": summary["overdue"], "tone": "late",
                "sub": f"{worst} days late" if summary["overdue"] == 1 else f"longest {worst} days late"}
               if summary["overdue"] else {"label": "Overdue", "value": 0, "sub": "nothing late"})

    fines = ({"label": "Fines", "value": f"KES {summary['fines_owed']}", "tone": "late", "sub": "pay at the desk"}
             if summary["fines_owed"] else {"label": "Fines", "value": "KES 0", "sub": "nothing owed"})

    if summary["borrowing_paused"]:
        fourth = {"label": "Borrowing", "value": "Paused", "tone": "late",
                  "sub": "until you return it" if summary["overdue"] == 1 else "until you return them"}
    elif on_time.empty:
        fourth = {"label": "Next due", "value": "None", "sub": "no books out"}
    else:
        first = on_time.iloc[0]
        fourth = {"label": "Next due", "value": dates.short(first["due_on"].date()),
                  "sub": f"{first['title'].split(':')[0]}, {days_left_text(int(first['days_left']))}"}
    return [on_loan, overdue, fines, fourth]


# --------------------------------------------------------------- wording
def days_left_text(days_left: int) -> str:
    if days_left < 0:
        return f"{plural(-days_left, 'day')} late"
    return {0: "today", 1: "tomorrow"}.get(days_left, f"in {days_left} days")


def due_line(row) -> tuple[str, str]:
    """('Due Fri 20 Nov · 4 days left', tone) for one open loan."""
    due = f"Due {dates.short(row.due_on.date())}"
    if row.days_left < 0:
        return f"{due} · {plural(-row.days_left, 'day')} late · KES {row.fine}", "late"
    if row.days_left == 0:
        return f"{due} · today", "soon"
    left = f"{plural(row.days_left, 'day')} left"
    return f"{due} · {left}", "soon" if row.status == "due soon" else ""


def availability_line(copies_free: int | None, outlook: dict | None = None) -> tuple[str, str]:
    """('3 copies free', 'ok') or ('Both copies out · next back Fri 20 Nov', 'soon')."""
    if copies_free is None:
        return "", ""
    if copies_free > 0:
        return f"{plural(copies_free, 'copy', 'copies')} free", "ok"
    outlook = outlook or {}
    out = outlook.get("copies_out", 0)
    which = "The copy is out" if out == 1 else ("Both copies out" if out == 2 else f"All {out} copies out")
    if outlook.get("next_due"):
        return f"{which} · next back {dates.short(outlook['next_due'])}", "soon"
    return f"{which} · overdue, no return date", "soon"


def hint(text: str | None) -> None:
    """The reason under a disabled button, in words (a greyed-out button alone explains nothing)."""
    if text:
        st.html(f'<div class="hint">{esc(text)}</div>')


# ------------------------------------------------------------------ book line
def book_line(title: str, author: str, genre: str, lines: list[tuple[str, str]] = (), width: int = 64,
              serif_title: bool = True) -> str:
    """A cover, the title, the author and a few short lines (text, tone), as one block of HTML.
    Tones: '' muted, 'ok' green, 'soon' amber, 'late' red, 'ink' normal text."""
    extra = "".join(f'<div class="bl {tone}">{esc(text)}</div>' for text, tone in lines if text)
    return (f'<div class="bk">{theme.cover(title, genre, author, width)}<div class="bi">'
            f'<div class="bt">{esc(title)}</div><div class="ba">{esc(author)}</div>{extra}</div></div>')


def alert_box(title: str, text: str) -> None:
    """The red box above My loans: what is wrong and what to do about it."""
    st.html(f'<div class="banner"><b>{esc(title)}</b><span>{esc(text)}</span></div>')


def overdue_banner(loans: pd.DataFrame) -> None:
    late = loans[loans["days_left"] < 0].sort_values("days_left")
    if late.empty:
        return
    first = late.iloc[0]
    if len(late) == 1:
        title = f"“{first['title']}” is {plural(-int(first['days_left']), 'day')} overdue."
    else:
        title = f"{len(late)} books are overdue, the longest by {plural(-int(first['days_left']), 'day')}."
    alert_box(title, f"Fine so far KES {int(late['fine'].sum())}, growing KES {FINE_PER_DAY} a day. "
                     "You can’t borrow or renew until you return "
                     f"{'it' if len(late) == 1 else 'them'}.")


def empty_state(title: str, text: str) -> None:
    st.html(f'<div class="empty"><b>{esc(title)}</b><span>{esc(text)}</span></div>')


def availability_map(conn) -> dict[int, int]:
    from library_agent import db
    frame = db.availability_df(conn)
    return {int(b): int(a) for b, a in zip(frame["book_id"], frame["available"])}


def outlooks(conn, book_ids, loans: pd.DataFrame | None = None) -> dict[int, dict]:
    """return_outlook for several books, loading the loans only once."""
    loans = circulation.open_loans(conn) if loans is None else loans
    return {int(b): circulation.return_outlook(conn, int(b), loans=loans) for b in book_ids}
