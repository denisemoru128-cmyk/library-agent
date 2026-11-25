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


def go(section: str) -> None:
    """Show another section. Used as a button's ``on_click``, so the screen changes on the next run."""
    st.session_state["section"] = section


def format_dates(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
    return frame