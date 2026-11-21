"""Streamlit interface: a student desk, a librarian desk and the library agent.

Run it with:  streamlit run app.py

Everything the interface does goes through the tested modules (circulation, agent,
chatbot, search, recommender, forecast). This file only shows results and passes clicks on.
"""
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from library_agent import circulation, db, report
from library_agent.agent import LibraryAgent
from library_agent.chatbot import LibraryChatbot
from library_agent.config import (
    DB_PATH, ESCALATION_DAYS, FORECAST_HORIZON_WEEKS, LOAN_DAYS, MAX_ACTIVE_LOANS,
)
from library_agent.forecast import DemandForecaster
from library_agent.recommender import Recommender
from library_agent.search import BookSearch

STUDENT_SECTIONS = ["Search", "For you", "My loans", "Notifications", "Ask the assistant"]
LIBRARIAN_SECTIONS = ["Dashboard", "Circulation desk", "Agent", "Demand & buying", "Catalogue", "Model quality"]
EXAMPLE_QUESTIONS = [
    "Find books about machine learning", "Is Clean Code available?", "What is due this week?",
    "How much do I owe in fines?", "Renew Clean Code", "Books similar to Atomic Habits",
    "What is the fine for late books?", "Which books need more copies?",
]


# ------------------------------------------------------------------ models
@dataclass
class Models:
    books: pd.DataFrame
    students: pd.DataFrame
    search: BookSearch
    recommender: Recommender
    forecaster: DemandForecaster
    chatbot: LibraryChatbot


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


# ----------------------------------------------------------------- helpers
def _flash(kind: str, text: str) -> None:
    """Keep a message for the next run. (A message shown right before st.rerun() would vanish.)"""
    st.session_state["_flash"] = (kind, text)


def _show_flash() -> None:
    kind, text = st.session_state.pop("_flash", (None, None))
    if kind:
        {"success": st.success, "error": st.error, "info": st.info, "warning": st.warning}[kind](text)


def _finish(outcome: circulation.Outcome) -> None:
    """Show the result of a circulation action (the rules decided it) and refresh the page."""
    _flash("success" if outcome.ok else "error", outcome.message)
    st.rerun()


def _nav(options: list[str], key: str) -> str:
    """Section switcher. Unlike st.tabs it remembers the choice after a refresh."""
    return st.segmented_control(
        "Section", options, default=options[0], key=key, label_visibility="collapsed") or options[0]


def _dates(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
    return frame


def _book_columns(**widths: int) -> dict:
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


def _borrow_form(conn, student_id: int, candidates: pd.DataFrame, key: str) -> None:
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
        _finish(circulation.issue_book(conn, student_id, int(book_id)))


# ------------------------------------------------------------ student view
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
                 hide_index=True, width="stretch", column_config=_book_columns())
    _borrow_form(conn, student_id, results, "search")


def _student_recommendations(conn, student_id: int, m: Models) -> None:
    st.caption("Mixes books similar to what you borrowed with what similar readers borrowed. "
               "New students get picks from their interests and popular titles.")
    picks = m.recommender.recommend(student_id, top_n=8, availability=db.availability_df(conn))
    st.dataframe(picks[["title", "author", "available", "score", "reason"]],
                 hide_index=True, width="stretch",
                 column_config=_book_columns(title=240, author=140, reason=360))
    _borrow_form(conn, student_id, picks, "recommended")


def _student_loans(conn, student_id: int, m: Models) -> None:
    loans = circulation.open_loans(conn, student_id)
    if loans.empty:
        st.info(f"You have no books on loan. You can borrow up to {MAX_ACTIVE_LOANS}.")
    else:
        late = loans[loans["status"] == "overdue"]
        if not late.empty:
            st.error(f"{len(late)} overdue book(s). Fine so far: KES {int(late['fine'].sum())}. "
                     "Return them to borrow or renew again.")
        view = _dates(loans, "due_on")[["title", "due_on", "days_left", "status", "fine", "renewals"]]
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={"title": st.column_config.TextColumn("title", width=300),
                                    "fine": "fine (KES)", "days_left": "days left"})
        labels = {int(row.loan_id): f"{row.title}  ·  due {row.due_on:%d %b}  ·  {row.status}"
                  for row in loans.itertuples()}
        loan_id = st.selectbox("Choose a loan", list(labels), format_func=labels.get)
        renew, give_back, _ = st.columns([1, 1, 4])
        if renew.button(f"Renew (+{LOAN_DAYS} days)"):
            _finish(circulation.renew_loan(conn, int(loan_id)))
        if give_back.button("Return"):
            _finish(circulation.return_book(conn, int(loan_id)))
    st.markdown("#### Borrow a book")
    available = db.availability_df(conn).merge(m.books[["book_id", "title"]], on="book_id")
    _borrow_form(conn, student_id, available.sort_values("title"), "any")


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
                     hide_index=True, width="stretch", column_config=_book_columns())
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


def _student_view(conn, student_id: int, m: Models) -> None:
    student = db.get_student(conn, student_id)
    summary = circulation.student_summary(conn, student_id)
    st.subheader(f"{student['name']}  ·  {student['reg_no']}")
    st.caption(f"{student['department']}  ·  interests: {str(student['interests']).replace('|', ', ')}")
    columns = st.columns(4)
    columns[0].metric("Books on loan", f"{summary['active_loans']} / {MAX_ACTIVE_LOANS}")
    columns[1].metric("Overdue", summary["overdue"])
    columns[2].metric("Fines owed", f"KES {summary['fines_owed']}")
    columns[3].metric("Can still borrow", summary["can_borrow_more"])
    _show_flash()
    section = _nav(STUDENT_SECTIONS, "student_section")
    {
        "Search": _student_search,
        "For you": _student_recommendations,
        "My loans": _student_loans,
        "Notifications": _student_notifications,
        "Ask the assistant": _student_chat,
    }[section](conn, student_id, m)


# --------------------------------------------------------- librarian view
def _librarian_dashboard(conn, m: Models) -> None:
    stats = circulation.library_stats(conn)
    columns = st.columns(5)
    columns[0].metric("Titles", stats["titles"])
    columns[1].metric("Copies on loan", f"{stats['copies_out']} / {stats['copies']}")
    columns[2].metric("Shelf utilisation", f"{stats['utilisation']:.0%}")
    columns[3].metric("Overdue loans", stats["overdue"])
    columns[4].metric("Fines outstanding", f"KES {stats['fines_outstanding']}")

    st.markdown("#### Loans per week: history and forecast")
    series = m.forecaster.total_series(horizon_weeks=FORECAST_HORIZON_WEEKS)
    wide = series.pivot(index="week_start", columns="kind", values="loans")
    last_actual = series.loc[series["kind"] == "actual", "week_start"].max()
    wide.loc[last_actual, "forecast"] = wide.loc[last_actual, "actual"]   # join the two lines
    st.line_chart(wide.tail(30))

    overdue = circulation.overdue_loans(conn).sort_values("days_overdue", ascending=False)
    st.markdown(f"#### Overdue ({len(overdue)})")
    st.dataframe(_dates(overdue, "due_on")[["student_name", "reg_no", "title", "due_on", "days_overdue", "fine"]],
                 hide_index=True, width="stretch", column_config={"fine": "fine (KES)"})
    soon = circulation.due_soon(conn)
    st.markdown(f"#### Due in the next few days ({len(soon)})")
    st.dataframe(_dates(soon, "due_on")[["student_name", "reg_no", "title", "due_on", "days_left"]],
                 hide_index=True, width="stretch")


def _librarian_desk(conn, m: Models) -> None:
    books = db.books_df(conn)
    issue, give_back = st.columns(2)
    with issue:
        st.markdown("#### Issue a book")
        students = {int(row.student_id): f"{row.reg_no}  ·  {row.name}" for row in m.students.itertuples()}
        free = db.availability_df(conn).merge(books[["book_id", "title"]], on="book_id")
        free = free[free["available"] > 0].sort_values("title")
        titles = {int(row.book_id): f"{row.title}  ({row.available} free)" for row in free.itertuples()}
        with st.form("issue_form"):
            student_id = st.selectbox("Student", list(students), format_func=students.get)
            book_id = st.selectbox("Book", list(titles), format_func=titles.get)
            submitted = st.form_submit_button("Issue")
        if submitted:
            _finish(circulation.issue_book(conn, int(student_id), int(book_id)))
    with give_back:
        st.markdown("#### Record a return")
        loans = circulation.open_loans(conn).sort_values("days_left")
        labels = {int(row.loan_id): f"{row.title}  —  {row.student_name}  ({row.status})"
                  for row in loans.itertuples()}
        with st.form("return_form"):
            loan_id = st.selectbox("Loan (most overdue first)", list(labels), format_func=labels.get)
            submitted = st.form_submit_button("Record return")
        if submitted:
            _finish(circulation.return_book(conn, int(loan_id)))


def _plan_tables(plan) -> None:
    """The agent's decisions, in detail."""
    if len(plan.escalations):
        st.markdown(f"**Final notices** (overdue more than {ESCALATION_DAYS} days)")
        st.dataframe(plan.escalations[["student_name", "reg_no", "title", "days_overdue", "fine"]],
                     hide_index=True, width="stretch")
    if len(plan.chases):
        st.markdown("**Priority chase**: overdue copies of books that are fully out and in demand")
        st.dataframe(plan.chases[["student_name", "title", "days_overdue", "forecast_loans", "advice"]],
                     hide_index=True, width="stretch")
    if len(plan.acquisitions):
        st.markdown("**Buy more copies**")
        st.dataframe(plan.acquisitions[["title", "copies", "forecast_loans", "copies_needed", "advice"]],
                     hide_index=True, width="stretch")


def _librarian_agent(conn, m: Models) -> None:
    as_of = db.today(conn)
    st.markdown(
        "The agent works in a loop: **perceive** (open loans, free copies, demand forecast), "
        "**decide** (reminders, overdue and final notices, buy proposals, priority chases) and "
        "**act** (send notices and write every decision, with its reason, to the log). "
        "Running it twice on the same day sends nothing twice.")
    preview_column, run_column, advance_column, _ = st.columns([1, 1, 1.4, 3])
    wants_preview = preview_column.button("Preview (dry run)")
    wants_run = run_column.button("Run agent now", type="primary")
    wants_next_day = advance_column.button("Advance date by 1 day")
    if wants_next_day:
        db.set_today(conn, as_of + timedelta(days=1))
        st.session_state.pop("agent_result", None)
        _flash("info", f"The library date is now {as_of + timedelta(days=1)}. Run the agent to see it react.")
        st.rerun()
    if wants_preview or wants_run:
        agent = LibraryAgent(conn, m.forecaster)
        with st.spinner("The agent is looking at the library..."):
            plan = agent.decide(agent.perceive())
            report = agent.act(plan) if wants_run else agent.preview(plan)
        st.session_state["agent_result"] = (report.summary(), plan)
        st.rerun()
    result = st.session_state.get("agent_result")
    if result:
        with st.container(border=True):
            st.markdown(result[0])
            _plan_tables(result[1])
    st.markdown("#### What the agent has done (its log)")
    log = db.agent_log_df(conn, limit=60)
    if len(log):
        st.dataframe(log[["run_on", "action", "detail"]], hide_index=True, width="stretch")
    else:
        st.caption("Nothing yet. Press “Run agent now”.")
    st.markdown("#### Messages sent to students")
    sent = db.notifications_df(conn, limit=60)
    if len(sent):
        st.dataframe(sent[["created_on", "student_name", "kind", "message"]], hide_index=True, width="stretch")


def _librarian_demand(conn, m: Models) -> None:
    demand = m.forecaster.book_demand(availability=db.availability_df(conn))
    to_buy = demand[demand["advice"].str.startswith("Buy")]
    columns = st.columns(3)
    columns[0].metric("Titles that need more copies", len(to_buy))
    columns[1].metric("Extra copies suggested", int((to_buy["copies_needed"] - to_buy["copies"]).sum()))
    columns[2].metric("On the watch list", int((demand["advice"] == "Watch").sum()))
    st.caption(f"A title needs more copies when its forecast loans over {FORECAST_HORIZON_WEEKS} weeks would keep "
               "its copies more than 70% busy. Sorted by demand per copy.")
    st.dataframe(demand.head(15)[["title", "copies", "available", "recent_loans",
                                  "forecast_loans", "copies_needed", "advice"]],
                 hide_index=True, width="stretch",
                 column_config={
                     "title": st.column_config.TextColumn("title", width=260),
                     "copies": st.column_config.NumberColumn("copies", width=70),
                     "available": st.column_config.NumberColumn("free now", width=80),
                     "recent_loans": st.column_config.NumberColumn("loans, last 8 wks", width=125),
                     "forecast_loans": st.column_config.NumberColumn("forecast, next 4 wks", width=140),
                     "copies_needed": st.column_config.NumberColumn("copies needed", width=105),
                     "advice": st.column_config.TextColumn("advice", width=130),
                 })
    st.markdown("#### Forecast detail")
    by = st.radio("Forecast level", ["genre", "book"], horizontal=True)
    forecast = m.forecaster.forecast(by=by, horizon_weeks=FORECAST_HORIZON_WEEKS)
    if by == "book":
        top = forecast.groupby("title")["predicted_loans"].sum().nlargest(8).index
        forecast = forecast[forecast["title"].isin(top)]
        st.caption("The 8 books with the highest forecast demand.")
    st.line_chart(forecast, x="week_start", y="predicted_loans", color="genre" if by == "genre" else "title")


def _librarian_catalogue(conn, m: Models) -> None:
    books = db.books_df(conn)
    with st.expander("Add a book to the catalogue"):
        with st.form("add_book_form", clear_on_submit=True):
            title = st.text_input("Title")
            author = st.text_input("Author")
            genre = st.selectbox("Genre", sorted(books["genre"].unique()))
            description = st.text_area("Short description (used by the search)")
            copies = st.number_input("Copies", min_value=1, max_value=50, value=2)
            submitted = st.form_submit_button("Add book")
        if submitted:
            if not (title.strip() and author.strip() and description.strip()):
                st.error("Please fill in the title, author and description.")
            elif title.strip().lower() in books["title"].str.lower().values:
                st.error("A book with this title is already in the catalogue.")
            else:
                db.add_book(conn, title, author, genre, description, int(copies))
                _flash("success", f"Added “{title.strip()}”. It is searchable straight away; "
                                  "press “Retrain models” in the sidebar to include it in the demand forecast.")
                st.rerun()
    catalogue = books.merge(db.availability_df(conn), on=["book_id", "copies"])
    st.dataframe(catalogue.drop(columns=["description"]), hide_index=True, width="stretch")


def _librarian_quality(conn, m: Models) -> None:
    results = report.load_results()
    if results is None:
        st.info("No results yet. Run `python evaluate.py` in the project folder (about 90 seconds), "
                "then reload this page.")
        return
    search, rec = results["search"], results["recommender"]["established students"]["methods"]
    genre, chat = results["forecast"]["genre"], results["chatbot"]["test"]
    columns = st.columns(4)
    columns[0].metric(f"Search hit rate @{search['k']}", f"{search['hit_rate']:.0%}")
    columns[1].metric("Recommender hit rate @10", f"{rec['hybrid']['hit_rate']:.0%}",
                      delta=f"{(rec['hybrid']['hit_rate'] - rec['popular']['hit_rate']) * 100:+.0f} pts vs most popular")
    columns[2].metric("Forecast error (genre WAPE)", f"{genre['model']['wape']:.1%}", delta_color="inverse",
                      delta=f"{(genre['model']['wape'] - genre['4-week average']['wape']) * 100:+.1f} pts vs 4-week average")
    columns[3].metric("Chatbot accuracy (held-out)", f"{chat['strict_accuracy']:.0%}")
    st.caption("Measured by `evaluate.py` on simulated data. Run it again after re-seeding to refresh these numbers.")
    st.markdown(report.render_markdown(results))


def _librarian_view(conn, m: Models) -> None:
    st.subheader("Librarian desk")
    _show_flash()
    section = _nav(LIBRARIAN_SECTIONS, "librarian_section")
    {
        "Dashboard": _librarian_dashboard,
        "Circulation desk": _librarian_desk,
        "Agent": _librarian_agent,
        "Demand & buying": _librarian_demand,
        "Catalogue": _librarian_catalogue,
        "Model quality": _librarian_quality,
    }[section](conn, m)


# -------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title="Library Agent", page_icon="📚", layout="wide")
    st.title("Library Agent")
    st.caption("AI-supported search, recommendations, library rules, demand forecasts and an agent that works on its own.")
    if not DB_PATH.exists():
        st.error("The library database is missing. Run `python seed.py` from the project directory first.")
        return

    conn = db.connect(DB_PATH)
    try:
        students = db.students_df(conn)
        if students.empty:
            st.error("No students are in the database. Seed the library before using the app.")
            return
        forecaster = load_forecaster(str(DB_PATH), st.session_state.get("model_version", 0))
        models = load_models(str(DB_PATH), DB_PATH.stat().st_mtime_ns, forecaster)

        with st.sidebar:
            st.markdown(f"**Library date:** {db.today(conn):%A, %d %B %Y}")
            role = st.radio("I am a", ["Student", "Librarian"], horizontal=True)
            student_id = None
            if role == "Student":
                names = {int(row.student_id): f"{row.name}  ({row.reg_no})" for row in students.itertuples()}
                student_id = st.selectbox("Student", list(names), format_func=names.get)
            st.divider()
            st.caption(f"Demand model trained on data up to {forecaster.as_of:%d %b %Y}.")
            if st.button("Retrain models"):
                st.session_state["model_version"] = st.session_state.get("model_version", 0) + 1
                st.rerun()
            st.caption("All people, books and loans in this demo are simulated.")

        if role == "Student":
            _student_view(conn, int(student_id), models)
        else:
            _librarian_view(conn, models)
    finally:
        conn.close()


if __name__ == "__main__":
    main()