"""Streamlit interface for the student and librarian library-agent views."""
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from library_agent import db
from library_agent.chatbot import LibraryChatbot
from library_agent.config import DB_PATH, FORECAST_HORIZON_WEEKS
from library_agent.forecast import DemandForecaster
from library_agent.recommender import Recommender
from library_agent.search import BookSearch


@st.cache_resource
def load_agent(database_path: str, database_mtime: int):
    """Cache model objects until the SQLite database changes."""
    del database_mtime
    conn = db.connect(Path(database_path))
    try:
        as_of = db.today(conn)
        books = db.books_df(conn)
        students = db.students_df(conn)
        loans = db.loans_df(conn)
    finally:
        conn.close()

    search = BookSearch(books)
    recommender = Recommender(books, students, loans, as_of)
    forecaster = DemandForecaster(books, loans, as_of)
    chatbot = LibraryChatbot(books, search, recommender, forecaster)
    return books, students, search, recommender, forecaster, chatbot


def _show_student_view(conn, student_id, books, search, recommender, chatbot):
    as_of = db.today(conn)
    availability = db.availability_df(conn)
    st.subheader("Student library")
    tabs = st.tabs(["Search", "Recommendations", "My loans", "Ask the agent"])

    with tabs[0]:
        query = st.text_input("Search by topic, title, or author", key="student_search")
        available_only = st.checkbox("Only show available copies", key="student_available_only")
        if query:
            results = search.search(
                query, top_n=10, availability=availability, available_only=available_only,
            )
            st.dataframe(results, hide_index=True, width="stretch")
            corrected = results.attrs.get("corrected_query")
            if corrected and corrected != query:
                st.caption(f"Interpreted search: {corrected}")

    with tabs[1]:
        recommendations = recommender.recommend(
            student_id, top_n=8, availability=availability,
        )
        st.dataframe(recommendations, hide_index=True, width="stretch")

    with tabs[2]:
        loans = db.student_loans_df(conn, student_id)
        if loans.empty:
            st.info("You have no active loans.")
        else:
            loans["overdue"] = loans["due_on"].dt.date < as_of
            overdue_count = int(loans["overdue"].sum())
            if overdue_count:
                st.warning(f"You have {overdue_count} overdue loan(s).")
            due_soon = (
                (loans["due_on"].dt.date >= as_of)
                & (loans["due_on"].dt.date <= as_of + timedelta(days=6 - as_of.weekday()))
            )
            if due_soon.any():
                st.info(f"You have {int(due_soon.sum())} loan(s) due this week.")
            st.dataframe(loans, hide_index=True, width="stretch")
            returnable = loans[["loan_id", "title"]].copy()
            returnable["label"] = returnable["loan_id"].astype(str) + " - " + returnable["title"]
            selection = st.selectbox("Select a book to return", returnable["label"].tolist())
            selected = returnable.loc[returnable["label"] == selection].iloc[0]
            if st.button("Record return", key="student_return"):
                db.record_return(conn, int(selected["loan_id"]), returned_on=as_of)
                st.success(f"Returned {selected['title']}.")
                st.rerun()

        available_books = availability.merge(books[["book_id", "title"]], on="book_id")
        available_books = available_books[available_books["available"] > 0]
        if available_books.empty:
            st.info("No copies are currently available to borrow.")
        else:
            with st.form("borrow_book_form"):
                choice = st.selectbox(
                    "Borrow an available title",
                    available_books["book_id"].tolist(),
                    format_func=lambda book_id: available_books.loc[
                        available_books["book_id"] == book_id, "title"
                    ].iloc[0],
                )
                submitted = st.form_submit_button("Borrow selected book")
            if submitted:
                try:
                    db.borrow_book(conn, student_id, int(choice), borrowed_on=as_of)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.success("Loan recorded.")
                    st.rerun()

    with tabs[3]:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        question = st.chat_input("Ask about a title, due dates, recommendations, or demand")
        if question:
            answer = chatbot.answer(question, conn, student_id=student_id)
            st.session_state.chat_history.append((question, answer))
        for question, answer in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer["message"])
                if answer["data"]:
                    st.dataframe(pd.DataFrame(answer["data"]), hide_index=True, width="stretch")


def _show_librarian_view(conn, books, forecaster):
    as_of = db.today(conn)
    availability = db.availability_df(conn)
    overdue = db.overdue_loans_df(conn, as_of=as_of)
    due_end = as_of + timedelta(days=6 - as_of.weekday())
    due_soon = db.due_loans_df(conn, start=as_of, end=due_end)

    st.subheader("Librarian dashboard")
    titles, copies = len(books), int(books["copies"].sum())
    out = int(availability["copies_out"].sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Titles", titles)
    col2.metric("Total copies", copies)
    col3.metric("Copies on loan", out)
    col4.metric("Overdue loans", len(overdue))

    tabs = st.tabs(["Due and overdue", "Demand forecast", "Availability"])
    with tabs[0]:
        st.markdown(f"#### Overdue ({len(overdue)})")
        st.dataframe(overdue, hide_index=True, width="stretch")
        st.markdown(f"#### Due by {due_end} ({len(due_soon)})")
        st.dataframe(due_soon, hide_index=True, width="stretch")

    with tabs[1]:
        by = st.radio("Forecast level", ["genre", "book"], horizontal=True)
        forecast = forecaster.forecast(by=by, horizon_weeks=FORECAST_HORIZON_WEEKS)
        st.line_chart(
            forecast,
            x="week_start",
            y="predicted_loans",
            color="genre" if by == "genre" else "title",
        )
        st.dataframe(forecast, hide_index=True, width="stretch")

    with tabs[2]:
        catalog = books.merge(availability, on=["book_id", "copies"])
        st.dataframe(catalog, hide_index=True, width="stretch")


def main():
    st.set_page_config(page_title="Library Agent", page_icon="📚", layout="wide")
    st.title("Library Agent")
    st.caption("AI-supported search, recommendations, demand forecasts, and library help.")
    if not DB_PATH.exists():
        st.error("The library database is missing. Run `python seed.py` from the project directory first.")
        return

    role = st.sidebar.radio("View", ["Student", "Librarian"])
    students_for_select = db.connect(DB_PATH)
    try:
        students = db.students_df(students_for_select)
    finally:
        students_for_select.close()
    if students.empty:
        st.error("No students are in the database. Seed the library before using the app.")
        return

    selected_student = None
    if role == "Student":
        selected_student = st.sidebar.selectbox(
            "Student",
            students["student_id"].tolist(),
            format_func=lambda student_id: students.loc[
                students["student_id"] == student_id, "name"
            ].iloc[0],
        )

    _, _, search, recommender, forecaster, chatbot = load_agent(
        str(DB_PATH), DB_PATH.stat().st_mtime_ns,
    )
    conn = db.connect(DB_PATH)
    try:
        if role == "Student":
            books = db.books_df(conn)
            _show_student_view(conn, int(selected_student), books, search, recommender, chatbot)
        else:
            books = db.books_df(conn)
            _show_librarian_view(conn, books, forecaster)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
