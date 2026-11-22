"""Pieces shared by the screens: the model bundle, flash messages and table helpers."""
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from library_agent import circulation
from library_agent.chatbot import LibraryChatbot
from library_agent.forecast import DemandForecaster
from library_agent.recommender import Recommender
from library_agent.search import BookSearch


@dataclass
class Models:
    books: pd.DataFrame
    students: pd.DataFrame
    search: BookSearch
    recommender: Recommender
    forecaster: DemandForecaster
    chatbot: LibraryChatbot


def flash(kind: str, text: str) -> None:
    """Keep a message for the next run. (A message shown right before st.rerun() would vanish.)"""
    st.session_state["_flash"] = (kind, text)


def show_flash() -> None:
    kind, text = st.session_state.pop("_flash", (None, None))
    if kind:
        {"success": st.success, "error": st.error, "info": st.info, "warning": st.warning}[kind](text)


def finish(outcome: circulation.Outcome) -> None:
    """Show the result of a circulation action (the rules decided it) and refresh the page."""
    flash("success" if outcome.ok else "error", outcome.message)
    st.rerun()


def _nav(options: list[str], key: str) -> str:
    """Section switcher. Unlike st.tabs it remembers the choice after a refresh."""
    return st.segmented_control(
        "Section", options, default=options[0], key=key, label_visibility="collapsed") or options[0]


def format_dates(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
    return frame


def book_columns(**widths: int) -> dict:
    """Column layout for tables of books, sized so the important columns fit without scrolling."""
    return {
        "title": st.column_config.TextColumn("title", width=widths.get("title", 300)),
        "author": st.column_config.TextColumn("author", width=widths.get("author", 170)),
        "genre": st.column_config.TextColumn("genre", width=150),
        "available": st.column_config.NumberColumn("free copies", width=90),
        "score": st.column_config.ProgressColumn("match", min_value=0.0, max_value=1.0, format="%.2f", width=95),
        "matched": st.column_config.TextColumn("matched words", width=120),
        "reason": st.column_config.TextColumn("why", width=widths.get("reason", 300)),
    }


def borrow_form(conn, student_id: int, candidates: pd.DataFrame, key: str) -> None:
    """Borrow one of the given titles (needs columns book_id, title, available)."""
    free = candidates[candidates["available"] > 0].drop_duplicates("book_id")
    if free.empty:
        st.info("None of these titles has a free copy right now.")
        return
    titles = dict(zip(free["book_id"].astype(int), free["title"]))
    with st.form(f"borrow_{key}"):
        book_id = st.selectbox("Borrow one of these titles", list(titles), format_func=titles.get)
        submitted = st.form_submit_button("Borrow")
    if submitted:
        finish(circulation.issue_book(conn, student_id, int(book_id)))