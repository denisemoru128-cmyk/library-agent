"""Streamlit interface: a student desk, a librarian desk and the library agent.

Run it with:  streamlit run app.py

Everything the interface does goes through the tested modules (circulation, agent,
chatbot, search, recommender, forecast). This file loads the models, handles sign-in and
picks the screen to show; the screens themselves live in the views/ package.
"""
from pathlib import Path

import streamlit as st

from library_agent import config, db
from library_agent.forecast import DemandForecaster
from library_agent.recommender import Recommender
from library_agent.chatbot import LibraryChatbot
from library_agent.search import BookSearch
from views import chat, librarian, shell, signin, student, theme
from views.common import Models, show_flash

MARK = Path(__file__).resolve().parent / "assets" / "karatina_mark.png"


# ------------------------------------------------------------------ models
@st.cache_resource(show_spinner="Training the demand forecast (about 15 seconds, only once)...")
def load_forecaster(database_path: str, version: int) -> DemandForecaster:
    """The slow model. Cached until the user presses "Retrain models" (which raises ``version``),
    so borrowing and returning books does not trigger a retrain."""
    del version
    conn = db.connect(Path(database_path))
    try:
        return DemandForecaster(db.books_df(conn), db.loans_df(conn), db.today(conn))
    finally:
        conn.close()


@st.cache_resource(max_entries=3)
def load_models(database_path: str, database_mtime: int, _forecaster: DemandForecaster) -> Models:
    """The fast models (a fraction of a second). Rebuilt whenever the database changes, so a new
    loan or a newly added book is seen immediately."""
    del database_mtime
    conn = db.connect(Path(database_path))
    try:
        as_of = db.today(conn)
        books, students, loans = db.books_df(conn), db.students_df(conn), db.loans_df(conn)
    finally:
        conn.close()
    search = BookSearch(books)
    recommender = Recommender(books, students, loans, as_of)
    chatbot = LibraryChatbot(books, search, recommender, _forecaster)
    return Models(books, students, search, recommender, _forecaster, chatbot)


# -------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title="Library Agent", page_icon=str(MARK), layout="wide")
    database = config.DB_PATH          # read at run time so tests can point the app at a copy
    user = st.session_state.get("user")
    theme.apply(hide_sidebar=user is None, chat_panel=bool(user and user["role"] == "student"))
    if not database.exists():
        st.error("The library database is missing. Run `python seed.py` from the project directory first.")
        return

    conn = db.connect(database)
    try:
        students = db.students_df(conn)
        if students.empty:
            st.error("No students are in the database. Seed the library before using the app.")
            return
        forecaster = load_forecaster(str(database), st.session_state.get("model_version", 0))
        models = load_models(str(database), database.stat().st_mtime_ns, forecaster)

        if user is None:
            signin.render(conn, students, models.books)
            return

        section = shell.sidebar(conn, user, forecaster)
        shell.top_row(conn, user, students)
        show_flash()
        if user["role"] == "student":
            student.SECTIONS[section](conn, user["student_id"], models)
            chat.render(conn, user["student_id"], models)      # docked on the right (below the page on a phone)
        else:
            librarian.SECTIONS[section](conn, models)
    finally:
        conn.close()


if __name__ == "__main__":
    main()