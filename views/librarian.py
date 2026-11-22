"""Librarian screens."""
from datetime import timedelta

import streamlit as st

from library_agent import circulation, db, report
from library_agent.agent import LibraryAgent
from library_agent.config import ESCALATION_DAYS, FORECAST_HORIZON_WEEKS

from .common import Models, flash, format_dates, finish


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
    st.dataframe(format_dates(overdue, "due_on")[["student_name", "reg_no", "title", "due_on", "days_overdue", "fine"]],
                 hide_index=True, width="stretch", column_config={"fine": "fine (KES)"})
    soon = circulation.due_soon(conn)
    st.markdown(f"#### Due in the next few days ({len(soon)})")
    st.dataframe(format_dates(soon, "due_on")[["student_name", "reg_no", "title", "due_on", "days_left"]],
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
            finish(circulation.issue_book(conn, int(student_id), int(book_id)))
    with give_back:
        st.markdown("#### Record a return")
        loans = circulation.open_loans(conn).sort_values("days_left")
        labels = {int(row.loan_id): f"{row.title}  —  {row.student_name}  ({row.status})"
                  for row in loans.itertuples()}
        with st.form("return_form"):
            loan_id = st.selectbox("Loan (most overdue first)", list(labels), format_func=labels.get)
            submitted = st.form_submit_button("Record return")
        if submitted:
            finish(circulation.return_book(conn, int(loan_id)))


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
        flash("info", f"The library date is now {as_of + timedelta(days=1)}. Run the agent to see it react.")
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
                flash("success", f"Added “{title.strip()}”. It is searchable straight away; "
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


SECTIONS = {
    "Dashboard": _librarian_dashboard,
    "Circulation desk": _librarian_desk,
    "Agent": _librarian_agent,
    "Demand & buying": _librarian_demand,
    "Catalogue": _librarian_catalogue,
    "Model quality": _librarian_quality,
}