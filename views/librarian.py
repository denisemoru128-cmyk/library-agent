"""Librarian screens: Dashboard (with the agent plan docked on the right), Circulation desk, Agent,
Demand & buying, Catalogue and Model quality.

Like the student screens, every number comes from the tested modules (circulation, forecast, agent, report).
"""
import altair as alt
import pandas as pd
import streamlit as st

from library_agent import circulation, dates, db, report
from library_agent.agent import LibraryAgent
from library_agent.config import ESCALATION_DAYS, FORECAST_HORIZON_WEEKS

from . import agent_panel
from . import components as ui
from .common import Models, finish, flash, format_dates, heading, section_title

DASHBOARD = "Dashboard"
FOREST, GOLD, MUTED = "#12372A", "#B8924A", "#5C6B63"
START_UP_WEEKS = 3        # the simulated library was still opening: its first weeks would flatten the chart
BUY_SHOWN = 5


# ------------------------------------------------------------------ Dashboard
def _loans_chart(series: pd.DataFrame, today) -> alt.Chart:
    """Loans per week: what happened (dark line) and what the model expects (dashed gold), with today marked."""
    series = series.assign(week_start=pd.to_datetime(series["week_start"]))
    actual = series[series["kind"] == "actual"].iloc[START_UP_WEEKS:]
    future = series[series["kind"] == "forecast"]
    joined = pd.concat([actual.tail(1).assign(kind="forecast"), future])        # so the two lines touch
    x = alt.X("week_start:T", title=None, axis=alt.Axis(format="%b", grid=False, labelColor=MUTED, tickCount=8))
    y = alt.Y("loans:Q", title=None, scale=alt.Scale(zero=True),
              axis=alt.Axis(gridColor="#E3E9E4", domain=False, ticks=False, labelColor=MUTED))
    today_at = pd.Timestamp(today)
    last = future.tail(1).iloc[0]
    average = round(float(future["loans"].mean()))
    labels = pd.DataFrame([
        {"week_start": actual["week_start"].iloc[len(actual) // 6], "loans": float(actual["loans"].iloc[:len(actual) // 3].max()) + 12,
         "text": "Actual loans", "colour": FOREST, "align": "left"},
        {"week_start": last["week_start"], "loans": float(last["loans"]), "text": f"Forecast: {average} a week",
         "colour": GOLD, "align": "left"},
        {"week_start": today_at, "loans": float(series["loans"].max()) * 1.08, "text": f"Today, {dates.short(today)}",
         "colour": MUTED, "align": "right"}])
    layers = [
        alt.Chart(actual).mark_line(color=FOREST, strokeWidth=2.4).encode(x=x, y=y),
        alt.Chart(joined).mark_line(color=GOLD, strokeWidth=2.4, strokeDash=[5, 4]).encode(x=x, y=y),
        alt.Chart(future.tail(1)).mark_point(color=GOLD, filled=True, size=70).encode(x=x, y=y),
        alt.Chart(pd.DataFrame({"week_start": [today_at]})).mark_rule(color="#8A9A90", strokeDash=[2, 3]).encode(x="week_start:T"),
        alt.Chart(labels).mark_text(dx=7, dy=-4, fontSize=13, fontWeight=600, align="left").encode(
            x="week_start:T", y="loans:Q", text="text:N", color=alt.Color("colour:N", scale=None))]
    return alt.layer(*layers).properties(height=250, padding={"right": 8, "top": 8})


def _buy_table(frame: pd.DataFrame) -> str:
    rows = "".join(
        f'<tr><td>{ui.esc(row.title)}</td><td class="n">{row.copies}</td><td class="n">{row.available}</td>'
        f'<td class="n">{row.forecast_loans:.1f}</td><td class="buy">{ui.esc(row.advice)}</td></tr>'
        for row in frame.itertuples())
    return ('<table class="tbl"><thead><tr><th>Title</th><th class="n">Owned</th><th class="n">Free now</th>'
            '<th class="n">Loans forecast, 4 weeks</th><th>Advice</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>")


def _dashboard(conn, m: Models) -> None:
    stats = circulation.library_stats(conn)
    overdue = circulation.overdue_loans(conn)
    serious = int((overdue["days_overdue"] > ESCALATION_DAYS).sum())
    series = m.forecaster.total_series(horizon_weeks=FORECAST_HORIZON_WEEKS)
    recent = series[series["kind"] == "actual"].tail(4)["loans"]
    average = round(float(recent.mean())) if len(recent) else 0

    n = stats["overdue"]
    heading("Library today", (f"{n} overdue {'loan needs' if n == 1 else 'loans need'} attention. " if n
                              else "Nothing is overdue. ") + "The agent’s plan is on the right.")
    ui.stat_strip([
        {"label": "Titles", "value": stats["titles"], "sub": f"{stats['copies']} copies"},
        {"label": "On loan", "value": stats["copies_out"], "unit": f"of {stats['copies']}",
         "sub": f"{stats['utilisation']:.0%} of copies"},
        {"label": "Overdue", "value": n, "tone": "late" if n else "", "sub": f"{serious} over {ESCALATION_DAYS} days"},
        {"label": "Fines owed", "value": f"KES {stats['fines_outstanding']:,}",
         "tone": "late" if stats["fines_outstanding"] else "", "sub": "on books still out"},
        {"label": "Loans, 7 days", "value": stats["loans_last_7_days"], "sub": f"about {average} a week"}])

    section_title("Loans per week")
    st.altair_chart(_loans_chart(series, db.today(conn)), width="stretch")
    st.caption(f"Forecast from the random forest model. The first {START_UP_WEEKS} weeks are not shown: "
               "the simulated library was still starting up.")

    demand = m.forecaster.book_demand(availability=db.availability_df(conn))
    to_buy = demand[demand["advice"].str.startswith("Buy")]
    section_title("Books to buy first", f"All {len(to_buy)} titles" if len(to_buy) > BUY_SHOWN else None,
                  "Demand & buying", "buy")
    if to_buy.empty:
        ui.empty_state("Nothing to buy", "No title is forecast to keep its copies more than 70% busy.")
    else:
        st.html(_buy_table(to_buy.head(BUY_SHOWN)))
    agent_panel.render(conn, m)


# ------------------------------------------------------------------ Circulation desk
def _table(headers: list[str], rows: list[list], numeric: tuple[int, ...] = (), late: tuple[int, ...] = ()) -> str:
    """A small read-only table. ``numeric`` columns are right-aligned, ``late`` columns are drawn in red."""
    def cell(i, value, tag="td"):
        classes = [name for name, on in (("n", i in numeric), ("late", tag == "td" and i in late)) if on]
        attribute = f' class="{" ".join(classes)}"' if classes else ""
        return f"<{tag}{attribute}>{ui.esc(value)}</{tag}>"
    head = "".join(cell(i, h, "th") for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(cell(i, v) for i, v in enumerate(row)) + "</tr>" for row in rows)
    return f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _overdue_table(overdue: pd.DataFrame) -> str:
    shown = format_dates(overdue, "due_on")
    return _table(["Student", "Reg. no.", "Title", "Was due", "Days late", "Fine (KES)"],
                  [[r.student_name, r.reg_no, r.title, r.due_on, r.days_overdue, r.fine] for r in shown.itertuples()],
                  numeric=(4, 5), late=(4, 5))


def _due_soon_table(soon: pd.DataFrame) -> str:
    shown = format_dates(soon.sort_values(["days_left", "student_name"]), "due_on")
    return _table(["Student", "Reg. no.", "Title", "Due", "Days left"],
                  [[r.student_name, r.reg_no, r.title, r.due_on, r.days_left] for r in shown.itertuples()],
                  numeric=(4,))


def _desk(conn, m: Models) -> None:
    heading("Circulation desk", "Issue and return books, and see who is late. The rules are the same ones the students see.")
    books = db.books_df(conn)
    issue, give_back = st.columns(2)
    with issue:
        st.markdown("### Issue a book")
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
        st.markdown("### Record a return")
        loans = circulation.open_loans(conn).sort_values("days_left")
        labels = {int(row.loan_id): f"{row.title}  —  {row.student_name}  ({row.status})"
                  for row in loans.itertuples()}
        with st.form("return_form"):
            loan_id = st.selectbox("Loan (most overdue first)", list(labels), format_func=labels.get)
            submitted = st.form_submit_button("Record return")
        if submitted:
            finish(circulation.return_book(conn, int(loan_id)))

    overdue = circulation.overdue_loans(conn)
    section_title(f"Overdue ({len(overdue)})")
    if overdue.empty:
        ui.empty_state("Nothing is overdue", "Every book out is within its loan period.")
    else:
        st.html(_overdue_table(overdue))
    soon = circulation.due_soon(conn)
    section_title(f"Due in the next few days ({len(soon)})")
    if soon.empty:
        ui.empty_state("Nothing is due soon", "No loan falls due in the next few days.")
    else:
        st.html(_due_soon_table(soon))


# ------------------------------------------------------------------ Agent
def _plan_tables(plan) -> None:
    """The agent's decisions, in detail."""
    if len(plan.escalations):
        st.markdown(f"**Final notices** (overdue more than {ESCALATION_DAYS} days)")
        st.html(_table(["Student", "Reg. no.", "Title", "Days late", "Fine (KES)"],
                       [[r.student_name, r.reg_no, r.title, r.days_overdue, r.fine] for r in plan.escalations.itertuples()],
                       numeric=(3, 4), late=(3, 4)))
    if len(plan.chases):
        st.markdown("**Chase first**: overdue copies of books that are fully out and in demand")
        st.html(_table(["Student", "Title", "Days late", "Loans forecast", "Advice"],
                       [[r.student_name, r.title, r.days_overdue, f"{r.forecast_loans:.1f}", r.advice]
                        for r in plan.chases.itertuples()],
                       numeric=(2, 3), late=(2,)))
    if len(plan.acquisitions):
        st.markdown("**Buy more copies**")
        st.html(_table(["Title", "Owned", "Loans forecast", "Copies needed", "Advice"],
                       [[r.title, r.copies, f"{r.forecast_loans:.1f}", r.copies_needed, r.advice]
                        for r in plan.acquisitions.itertuples()],
                       numeric=(1, 2, 3)))


def _agent(conn, m: Models) -> None:
    heading("Agent", "It looks at the library, decides what needs doing and does it, once a day.")
    st.markdown(
        "The agent works in a loop: **perceive** (open loans, free copies, demand forecast), "
        "**decide** (reminders, overdue and final notices, buy proposals, priority chases) and "
        "**act** (send notices and write every decision, with its reason, to the log). "
        "Running it twice on the same day sends nothing twice.")
    left, right = st.columns([2, 3], vertical_alignment="center")
    with left:
        agent_panel.run_buttons(conn, m, "agent")
    with right, st.container(key="agent_clock"):
        agent_panel.clock_buttons(conn, "agent")
    result = st.session_state.get("agent_result")
    if result:
        with st.container(border=True, key="agent_result_box"):
            st.markdown(result["summary"])
            agent = LibraryAgent(conn, m.forecaster)
            _plan_tables(agent.decide(agent.perceive()))
    st.markdown("### What the agent has done (its log)")
    log = db.agent_log_df(conn, limit=60)
    if len(log):
        st.dataframe(format_dates(log, "run_on")[["run_on", "action", "detail"]], hide_index=True, width="stretch",
                     column_config={"run_on": "Date", "action": "Action", "detail": "Detail"})
    else:
        ui.empty_state("Nothing yet", "Press “Run agent” to make the first entry.")
    st.markdown("### Messages sent to students")
    sent = db.notifications_df(conn, limit=60)
    if len(sent):
        st.dataframe(format_dates(sent, "created_on")[["created_on", "student_name", "kind", "message"]],
                     hide_index=True, width="stretch",
                     column_config={"created_on": "Date", "student_name": "Student", "kind": "Kind", "message": "Message"})
    else:
        ui.empty_state("No messages yet", "Notices appear here once the agent has run.")


# ------------------------------------------------------------------ Demand & buying
def _demand(conn, m: Models) -> None:
    heading("Demand & buying", f"Forecast loans for the next {FORECAST_HORIZON_WEEKS} weeks, and which titles need more copies.")
    demand = m.forecaster.book_demand(availability=db.availability_df(conn))
    to_buy = demand[demand["advice"].str.startswith("Buy")]
    ui.stat_strip([
        {"label": "Titles that need more copies", "value": len(to_buy)},
        {"label": "Extra copies suggested", "value": int((to_buy["copies_needed"] - to_buy["copies"]).sum())},
        {"label": "On the watch list", "value": int((demand["advice"] == "Watch").sum())}])
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
    st.markdown("### Forecast detail")
    by = st.radio("Forecast level", ["genre", "book"], horizontal=True)
    forecast = m.forecaster.forecast(by=by, horizon_weeks=FORECAST_HORIZON_WEEKS)
    if by == "book":
        top = forecast.groupby("title")["predicted_loans"].sum().nlargest(8).index
        forecast = forecast[forecast["title"].isin(top)]
        st.caption("The 8 books with the highest forecast demand.")
    st.line_chart(forecast, x="week_start", y="predicted_loans", color="genre" if by == "genre" else "title")


# ------------------------------------------------------------------ Catalogue
def _catalogue(conn, m: Models) -> None:
    heading("Catalogue", "Every title, its copies and how many are free.")
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


# ------------------------------------------------------------------ Model quality
def _quality(conn, m: Models) -> None:
    heading("Model quality", "How well the search, recommender, forecast and chatbot do on data they did not train on.")
    results = report.load_results()
    if results is None:
        ui.empty_state("No results yet", "Run `python evaluate.py` in the project folder (about 90 seconds), "
                                         "then reload this page.")
        return
    search, rec = results["search"], results["recommender"]["established students"]["methods"]
    genre, chat = results["forecast"]["genre"], results["chatbot"]["test"]
    ui.stat_strip([
        {"label": f"Search hit rate @{search['k']}", "value": f"{search['hit_rate']:.0%}"},
        {"label": "Recommender hit rate @10", "value": f"{rec['hybrid']['hit_rate']:.0%}",
         "sub": f"{(rec['hybrid']['hit_rate'] - rec['popular']['hit_rate']) * 100:+.0f} pts vs most popular"},
        {"label": "Forecast error (genre WAPE)", "value": f"{genre['model']['wape']:.1%}",
         "sub": f"{(genre['model']['wape'] - genre['4-week average']['wape']) * 100:+.1f} pts vs 4-week average"},
        {"label": "Chatbot accuracy", "value": f"{chat['strict_accuracy']:.0%}", "sub": "held-out questions"}])
    st.caption("Measured by `evaluate.py` on simulated data. Run it again after re-seeding to refresh these numbers.")
    st.markdown(report.render_markdown(results))


SECTIONS = {
    DASHBOARD: _dashboard,
    "Circulation desk": _desk,
    "Agent": _agent,
    "Demand & buying": _demand,
    "Catalogue": _catalogue,
    "Model quality": _quality,
}