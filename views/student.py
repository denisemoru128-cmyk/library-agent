"""Student screens."""
import pandas as pd
import streamlit as st

from library_agent import circulation, db
from library_agent.config import LOAN_DAYS, MAX_ACTIVE_LOANS

from .common import Models, book_columns, borrow_form, finish, format_dates

EXAMPLE_QUESTIONS = [
    "Find books about machine learning", "Is Clean Code available?", "What is due this week?",
    "How much do I owe in fines?", "Renew Clean Code", "Books similar to Atomic Habits",
    "What is the fine for late books?", "Which books need more copies?",
]


def _student_search(conn, student_id: int, m: Models) -> None:
    query = st.text_input("Search by topic, title or author", key="student_search",
                          placeholder="for example: machin lerning, or history of kenya")
    available_only = st.checkbox("Only titles with a free copy", key="student_available_only")
    if not query:
        st.caption("Typos are corrected and related words are understood (TF-IDF search).")
        return
    results = m.search.search(query, top_n=10, availability=db.availability_df(conn),
                              available_only=available_only)
    corrected = results.attrs.get("corrected_query")
    if corrected and corrected != query:
        st.caption(f"Searching for: {corrected}")
    if results.empty:
        st.info("No matching titles. Try different words.")
        return
    st.dataframe(results[["title", "author", "genre", "available", "score", "matched"]],
                 hide_index=True, width="stretch", column_config=book_columns())
    borrow_form(conn, student_id, results, "search")


def _student_recommendations(conn, student_id: int, m: Models) -> None:
    st.caption("Mixes books similar to what you borrowed with what similar readers borrowed. "
               "New students get picks from their interests and popular titles.")
    picks = m.recommender.recommend(student_id, top_n=8, availability=db.availability_df(conn))
    st.dataframe(picks[["title", "author", "available", "score", "reason"]],
                 hide_index=True, width="stretch",
                 column_config=book_columns(title=240, author=140, reason=360))
    borrow_form(conn, student_id, picks, "recommended")


def _student_loans(conn, student_id: int, m: Models) -> None:
    loans = circulation.open_loans(conn, student_id)
    if loans.empty:
        st.info(f"You have no books on loan. You can borrow up to {MAX_ACTIVE_LOANS}.")
    else:
        late = loans[loans["status"] == "overdue"]
        if not late.empty:
            st.error(f"{len(late)} overdue book(s). Fine so far: KES {int(late['fine'].sum())}. "
                     "Return them to borrow or renew again.")
        view = format_dates(loans, "due_on")[["title", "due_on", "days_left", "status", "fine", "renewals"]]
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={"title": st.column_config.TextColumn("title", width=300),
                                    "fine": "fine (KES)", "days_left": "days left"})
        labels = {int(row.loan_id): f"{row.title}  ·  due {row.due_on:%d %b}  ·  {row.status}"
                  for row in loans.itertuples()}
        loan_id = st.selectbox("Choose a loan", list(labels), format_func=labels.get)
        renew, give_back, _ = st.columns([1, 1, 4])
        if renew.button(f"Renew (+{LOAN_DAYS} days)"):
            finish(circulation.renew_loan(conn, int(loan_id)))
        if give_back.button("Return"):
            finish(circulation.return_book(conn, int(loan_id)))
    st.markdown("#### Borrow a book")
    available = db.availability_df(conn).merge(m.books[["book_id", "title"]], on="book_id")
    borrow_form(conn, student_id, available.sort_values("title"), "any")


def _student_notifications(conn, student_id: int, m: Models) -> None:
    messages = db.notifications_df(conn, student_id, limit=50)
    if messages.empty:
        st.info("No messages yet. The library agent sends reminders when it runs "
                "(Librarian view, Agent section).")
        return
    for row in messages.itertuples():
        show = st.error if row.kind in {"overdue", "final_notice"} else st.info
        show(f"{row.message}", icon="⏰" if row.kind == "due_soon" else "⚠️")
        st.caption(f"{row.created_on} · {row.kind.replace('_', ' ')}")


def _render_answer(answer: dict) -> None:
    st.markdown(answer["message"])
    if answer["data"]:
        frame = pd.DataFrame(answer["data"])
        st.dataframe(frame.drop(columns=[c for c in frame.columns if c.endswith("_id")]),
                     hide_index=True, width="stretch", column_config=book_columns())
    st.caption(f"understood as: {answer['intent']} · confidence {answer['confidence']:.0%}")


def _student_chat(conn, student_id: int, m: Models) -> None:
    history = st.session_state.setdefault("chat_histories", {}).setdefault(student_id, [])
    context = st.session_state.setdefault("chat_contexts", {}).setdefault(student_id, {})
    with st.expander("What can I ask?"):
        st.markdown("\n".join(f"- {example}" for example in EXAMPLE_QUESTIONS))
        st.caption("I remember the last book we talked about, so you can follow up with “is it available?”.")
    for question, answer in history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            _render_answer(answer)
    question = st.chat_input("Ask about a book, your loans, the library rules or recommendations")
    if question:
        answer = m.chatbot.answer(question, conn, student_id=student_id, context=context)
        history.append((question, answer))
        st.rerun()  # also refreshes the numbers at the top, which a renewal may have changed


SECTIONS = {
    "Search": _student_search,
    "For you": _student_recommendations,
    "My loans": _student_loans,
    "Notifications": _student_notifications,
    "Assistant": _student_chat,
}