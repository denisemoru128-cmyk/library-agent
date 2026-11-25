"""Pieces shared by the screens: the model bundle, flash messages, headings and table helpers."""
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from library_agent import circulation, dates
from library_agent.chatbot import LibraryChatbot
from library_agent.forecast import DemandForecaster
from library_agent.recommender import Recommender
from library_agent.search import BookSearch

from . import components as ui


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


def go(section: str) -> None:
    """Show another section. Used as a button's ``on_click``, so the screen changes on the next run."""
    st.session_state["section"] = section


def heading(title: str, sub: str = "") -> None:
    """The page title, and one sentence under it."""
    st.markdown(f"# {title}")
    if sub:
        st.html(f'<p class="sub">{ui.esc(sub)}</p>')


def section_title(title: str, link: str | None = None, target: str | None = None, key: str = "") -> None:
    """A smaller title with an optional link on the right that opens another section."""
    left, right = st.columns([5, 2], vertical_alignment="center")
    left.markdown(f"### {title}")
    if link:
        right.button(link, key=f"link_{key}", type="tertiary", on_click=go, args=(target,))


def format_dates(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Dates for people, with the weekday: 'Sat 21 Nov'."""
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_datetime(frame[column]).map(lambda day: dates.short(day.date()))
    return frame