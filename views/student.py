"""Student screens: Home, Search, My loans and For you. (The assistant lives in views/chat.py.)

Every button that changes the library asks library_agent.circulation to do it. The buttons that are switched
off ask the same module *why* (circulation.borrow_blocker / renew_blocker) and print the reason under them.
"""
import pandas as pd
import streamlit as st

from library_agent import circulation, db
from library_agent.config import MAX_ACTIVE_LOANS

from . import components as ui
from .common import Models, finish, go

SEARCH_EXAMPLES = ["machine learning", "history of kenya", "statistics", "contract law"]


# ------------------------------------------------------------------ shared pieces
def _position(conn, student_id: int) -> tuple[dict, pd.DataFrame]:
    return circulation.student_summary(conn, student_id), circulation.open_loans(conn, student_id)


def _standing(summary: dict) -> str:
    """One sentence under the greeting: what the student has and what they can do."""
    if summary["active_loans"] == 0:
        return f"You have no books on loan. You can borrow up to {MAX_ACTIVE_LOANS}."
    have = f"{ui.plural(summary['active_loans'], 'book')} on loan."
    if summary["borrowing_paused"]:
        return f"{have} Borrowing is paused until you return “{summary['blocking_title']}”."
    if summary["slots_free"] == 0:
        return f"{have} That is the limit of {MAX_ACTIVE_LOANS}. Return one to borrow another."
    return f"{have} You can borrow {summary['slots_free']} more."


def _heading(title: str, sub: str = "") -> None:
    st.markdown(f"# {title}")
    if sub:
        st.html(f'<p class="sub">{ui.esc(sub)}</p>')


def _section_title(title: str, link: str | None = None, target: str | None = None, key: str = "") -> None:
    left, right = st.columns([5, 2], vertical_alignment="center")
    left.markdown(f"### {title}")
    if link:
        right.button(link, key=f"link_{key}", type="tertiary", on_click=go, args=(target,))


def _renew_button(conn, loans: pd.DataFrame, loan_id: int, key: str) -> None:
    reason = circulation.renew_blocker(loans, loan_id)
    if st.button("Renew", key=key, disabled=bool(reason)):
        finish(circulation.renew_loan(conn, loan_id))
    ui.hint(reason)


def _return_button(conn, loan_id: int, key: str, primary: bool) -> None:
    if st.button("Return", key=key, type="primary" if primary else "secondary"):
        finish(circulation.return_book(conn, loan_id))


def _borrow_button(conn, student_id: int, loans: pd.DataFrame, book_id: int, free: int, key: str) -> None:
    reason = circulation.borrow_blocker(loans, book_id, free)
    if st.button("Borrow", key=key, disabled=bool(reason)):
        finish(circulation.issue_book(conn, student_id, book_id))
    ui.hint(reason)


def _loan_card(conn, loans: pd.DataFrame, row, key: str) -> None:
    text, tone = ui.due_line(row)
    with st.container(key=f"card_{key}_{row.loan_id}"):
        st.html(ui.book_line(row.title, row.author, row.genre, [(text, tone)], width=76))
        if row.status == "overdue":
            _return_button(conn, int(row.loan_id), f"{key}_return_{row.loan_id}", primary=True)
        else:
            _renew_button(conn, loans, int(row.loan_id), f"{key}_renew_{row.loan_id}")


def _book_lines(pick, free: int | None, outlook: dict | None) -> list[tuple[str, str]]:
    lines = []
    reason = getattr(pick, "reason", "")
    if reason:
        lines.append((reason, "ink"))
    lines.append(ui.availability_line(free, outlook))
    return lines


def _pick_card(conn, student_id: int, loans: pd.DataFrame, pick, free: int, outlook, key: str) -> None:
    with st.container(key=f"card_{key}_{pick.book_id}"):
        st.html(ui.book_line(pick.title, pick.author, pick.genre, _book_lines(pick, free, outlook), width=76))
        _borrow_button(conn, student_id, loans, int(pick.book_id), free, f"{key}_borrow_{pick.book_id}")


def _book_rows(conn, student_id: int, loans: pd.DataFrame, frame: pd.DataFrame, key: str,
               extra_line=None) -> None:
    """One row per book: cover, title, author, why it is shown, availability, and a Borrow button."""
    free = ui.availability_map(conn)
    all_loans = circulation.open_loans(conn)
    out = ui.outlooks(conn, [int(b) for b in frame["book_id"] if free.get(int(b), 0) == 0], all_loans)
    for pick in frame.itertuples():
        book_id = int(pick.book_id)
        lines = _book_lines(pick, free.get(book_id), out.get(book_id))
        if extra_line:
            lines.append((extra_line(pick), ""))
        with st.container(key=f"row_{key}_{book_id}"):
            left, right = st.columns([7, 2], vertical_alignment="center")
            left.html(ui.book_line(pick.title, pick.author, pick.genre, lines, width=52))
            with right:
                _borrow_button(conn, student_id, loans, book_id, free.get(book_id, 0), f"{key}_borrow_{book_id}")


# ------------------------------------------------------------------ Home
def _home(conn, student_id: int, m: Models) -> None:
    summary, loans = _position(conn, student_id)
    name = m.students.loc[m.students["student_id"] == student_id, "name"].iloc[0]
    _heading(f"Welcome back, {name.split()[0]}", _standing(summary))
    ui.stat_strip(ui.student_tiles(summary, loans))

    _section_title("Your loans", "All loans", "My loans", "loans")
    if loans.empty:
        ui.empty_state("No books on loan", "Find something in Search, or look at For you.")
    else:
        for column, row in zip(st.columns(3), loans.sort_values("days_left").head(3).itertuples()):
            with column:
                _loan_card(conn, loans, row, "home")

    _section_title("For you", "All recommendations", "For you", "reco")
    free = ui.availability_map(conn)
    picks = m.recommender.recommend(student_id, top_n=3, availability=db.availability_df(conn))
    out = ui.outlooks(conn, [int(b) for b in picks["book_id"] if free.get(int(b), 0) == 0])
    for column, pick in zip(st.columns(3), picks.itertuples()):
        with column:
            _pick_card(conn, student_id, loans, pick, free.get(int(pick.book_id), 0), out.get(int(pick.book_id)), "home")


# ------------------------------------------------------------------ Search
def _set_query(text: str) -> None:
    st.session_state["search_query"] = text
    st.session_state["top_search"] = text


def _search(conn, student_id: int, m: Models) -> None:
    _heading("Search")
    query = st.session_state.get("search_query", "").strip()
    only_free = st.checkbox("Only titles with a free copy", key="student_available_only")
    if not query:
        ui.empty_state("Search the catalogue",
                       "Type a topic, title or author in the box at the top. Typos are corrected and "
                       "related words are understood.")
        st.caption("Try one of these:")
        for column, example in zip(st.columns(len(SEARCH_EXAMPLES)), SEARCH_EXAMPLES):
            column.button(example, key=f"try_{example}", on_click=_set_query, args=(example,))
        return
    results = m.search.search(query, top_n=10, availability=db.availability_df(conn), available_only=only_free)
    corrected = results.attrs.get("corrected_query")
    if results.empty:
        ui.empty_state("No matching titles", "Try different words, or untick “Only titles with a free copy”."
                       if only_free else "Try different words.")
        return
    shown = f"{ui.plural(len(results), 'title')} for “{corrected or query}”"
    st.caption(shown + (f" (you typed “{query}”)" if corrected and corrected != query else ""))
    _, loans = _position(conn, student_id)
    _book_rows(conn, student_id, loans, results, "search",
               extra_line=lambda pick: f"{pick.score:.0%} match" + (f" · matched on: {pick.matched}" if pick.matched else ""))


# ------------------------------------------------------------------ My loans
def _due_block(row) -> str:
    text, tone = ui.due_line(row)
    due, _, rest = text.partition(" · ")
    return f'<div class="due"><b>{ui.esc(due)}</b><small class="{tone}">{ui.esc(rest)}</small></div>'


def _my_loans(conn, student_id: int, m: Models) -> None:
    summary, loans = _position(conn, student_id)
    _heading("My loans")
    ui.overdue_banner(loans)
    ui.stat_strip(ui.student_tiles(summary, loans))
    if loans.empty:
        ui.empty_state("No books on loan", f"You can borrow up to {MAX_ACTIVE_LOANS}.")
        search, picks = st.columns([1, 1])
        search.button("Search the catalogue", key="empty_search", on_click=go, args=("Search",))
        picks.button("See books for you", key="empty_reco", on_click=go, args=("For you",))
        return
    st.markdown("### Books on loan")
    for row in loans.sort_values("days_left").itertuples():
        late = row.status == "overdue"
        with st.container(key=f"{'lrow' if late else 'row'}_loan_{row.loan_id}"):
            book, due, renew, back = st.columns([5, 3, 2, 2])
            book.html(ui.book_line(row.title, row.author, row.genre, width=52))
            due.html(_due_block(row))
            with renew:
                if not late:   # an overdue book can only be returned; the red box above says so
                    _renew_button(conn, loans, int(row.loan_id), f"loans_renew_{row.loan_id}")
            with back:
                _return_button(conn, int(row.loan_id), f"loans_return_{row.loan_id}", primary=late)


# ------------------------------------------------------------------ For you
def _for_you(conn, student_id: int, m: Models) -> None:
    _heading("For you", "Books like the ones you borrowed, plus what similar readers borrowed. "
                        "New students get picks from their interests and popular titles.")
    subject = st.columns([2, 5])[0].selectbox(
        "Subject", ["All subjects"] + sorted(m.books["genre"].unique()), key="reco_genre")
    picks = m.recommender.recommend(student_id, top_n=8, availability=db.availability_df(conn),
                                    genre=None if subject == "All subjects" else subject)
    if picks.empty:
        ui.empty_state("Nothing new in this subject", "You have already borrowed every title here.")
        return
    _, loans = _position(conn, student_id)
    _book_rows(conn, student_id, loans, picks, "reco")


SECTIONS = {
    "Home": _home,
    "Search": _search,
    "My loans": _my_loans,
    "For you": _for_you,
}