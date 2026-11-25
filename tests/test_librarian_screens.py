"""The librarian screens: Dashboard (numbers, chart, buy list, docked agent plan), Circulation desk and Agent.

The real app is run with AppTest against a private copy of the database, so these check that every number on
screen comes from the tested modules and that the buttons change the library the way the rules say.
"""
from datetime import date, timedelta

from library_agent import circulation, db, dates
from views import librarian

from .conftest import AS_OF


def _page(at) -> str:
    """All the HTML the screen drew (st.html), as one string."""
    return " ".join(h.value for h in at.get("html"))


def _headings(at) -> str:
    """All the markdown the screen drew (page and section titles), as one string."""
    return " ".join(m.value for m in at.markdown)


def _keys(at) -> set[str]:
    return {b.key for b in at.button}


def _count(at, table: str) -> int:
    conn = db.connect(at.db)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def _library_date(at) -> date:
    conn = db.connect(at.db)
    try:
        return db.today(conn)
    finally:
        conn.close()


# ------------------------------------------------------------------ Dashboard
def test_dashboard_strip_shows_the_numbers_from_the_library(app, conn):
    stats = circulation.library_stats(conn)
    at = app(role="Librarian")
    page = _page(at)

    assert "# Library today" in _headings(at)
    for label in ("Titles", "On loan", "Overdue", "Fines owed", "Loans, 7 days"):
        assert label in page
    assert f"KES {stats['fines_outstanding']:,}" in page
    assert f"of {stats['copies']}" in page


def test_dashboard_says_how_many_loans_are_overdue(app, conn):
    n = circulation.library_stats(conn)["overdue"]
    sentence = "1 overdue loan needs attention" if n == 1 else f"{n} overdue loans need attention"

    assert sentence in _page(app(role="Librarian"))


def test_buy_list_shows_only_the_top_five_titles_that_need_copies(app, conn, forecaster):
    demand = forecaster.book_demand(availability=db.availability_df(conn))
    wanted = list(demand[demand["advice"].str.startswith("Buy")]["title"].head(librarian.BUY_SHOWN))
    page = _page(app(role="Librarian"))

    table = page[page.index('<table class="tbl">'):]
    table = table[:table.index("</table>")]
    assert table.count("<tr>") == len(wanted) + 1                      # a header row and one row per title
    assert all(title.replace("&", "&amp;") in table for title in wanted)
    assert "Buy" in table


def test_loans_chart_drops_the_start_up_weeks_and_marks_today(forecaster):
    series = forecaster.total_series(horizon_weeks=4)
    spec = librarian._loans_chart(series, AS_OF).to_dict()
    data = spec["datasets"]
    first_layer = data[spec["layer"][0]["data"]["name"]]

    assert len(first_layer) == int((series["kind"] == "actual").sum()) - librarian.START_UP_WEEKS
    assert f"Today, {dates.short(AS_OF)}" in str(spec)
    assert "Forecast:" in str(spec)


def test_dashboard_draws_the_loans_chart(app):
    at = app(role="Librarian")

    assert len(at.get("vega_lite_chart")) == 1


# ------------------------------------------------------------------ the agent plan, docked on the Dashboard
def test_the_plan_panel_is_only_on_the_dashboard(app):
    assert {"plan_preview", "plan_run"} <= _keys(app(role="Librarian"))
    for section in ("Circulation desk", "Agent", "Catalogue"):
        assert "plan_run" not in _keys(app(role="Librarian", section=section))


def test_the_plan_panel_names_the_day_and_starts_as_a_preview(app):
    page = _page(app(role="Librarian"))

    assert f"Plan for {dates.full(AS_OF)}" in page
    assert "Preview only" in page
    assert "Final notices" in page and "Chase first" in page


def test_plan_preview_changes_nothing_and_run_sends_the_notices(app):
    at = app(role="Librarian")
    at.button(key="plan_preview").click().run()
    assert _count(at, "agent_log") == 0 and _count(at, "notifications") == 0
    assert "Preview only" in _page(at)

    at.button(key="plan_run").click().run()
    assert not at.exception
    assert _count(at, "agent_log") > 0 and _count(at, "notifications") > 0
    assert "Done." in _page(at)


def test_running_the_agent_from_the_panel_twice_sends_nothing_twice(app):
    at = app(role="Librarian")
    at.button(key="plan_run").click().run()
    sent, logged = _count(at, "notifications"), _count(at, "agent_log")

    at.button(key="plan_run").click().run()

    assert (_count(at, "notifications"), _count(at, "agent_log")) == (sent, logged)
    assert "Already run today" in _page(at)


def test_the_panel_lists_the_longest_overdue_loans_as_final_notices(app, conn):
    from library_agent.config import ESCALATION_DAYS
    serious = circulation.overdue_loans(conn)
    serious = serious[serious["days_overdue"] > ESCALATION_DAYS]
    page = _page(app(role="Librarian"))

    assert f'<span class="count">{len(serious)}</span>' in page
    if len(serious):
        assert serious.iloc[0]["student_name"] in page                  # the worst case is listed first


def test_advance_one_day_moves_the_library_clock_and_clears_the_result(app):
    at = app(role="Librarian")
    at.button(key="plan_run").click().run()
    at.button(key="plan_advance").click().run()

    assert _library_date(at) == AS_OF + timedelta(days=1)
    assert f"Plan for {dates.full(AS_OF + timedelta(days=1))}" in _page(at)
    assert "Preview only" in _page(at)                                   # yesterday's result is not shown today
    assert at.info[0].value.startswith("The library date is now")


def test_set_date_to_today_brings_a_stale_clock_up_to_the_real_date(app):
    at = app(role="Librarian")
    assert "plan_today" in _keys(at)                                     # the test library is set to 16 Nov

    at.button(key="plan_today").click().run()

    assert _library_date(at) == date.today()
    assert "plan_today" not in _keys(at)                                 # nothing to fix any more
    assert f"Plan for {dates.full(date.today())}" in _page(at)


# ------------------------------------------------------------------ Circulation desk
def test_desk_lists_every_overdue_loan_with_its_fine(app, conn):
    overdue = circulation.overdue_loans(conn)
    at = app(role="Librarian", section="Circulation desk")
    page = _page(at)

    assert f"### Overdue ({len(overdue)})" in _headings(at)
    first = overdue.iloc[0]
    assert first["student_name"] in page and f'<td class="n late">{first["fine"]}</td>' in page


def test_desk_shows_dates_with_their_weekday(app, conn):
    due = circulation.due_soon(conn).iloc[0]["due_on"].date()
    assert dates.short(due) in _page(app(role="Librarian", section="Circulation desk"))


def test_recording_a_return_from_the_desk_closes_the_most_overdue_loan(app, conn):
    before = len(circulation.open_loans(conn))
    worst = circulation.overdue_loans(conn).iloc[0]
    at = app(role="Librarian", section="Circulation desk")
    next(b for b in at.button if b.label == "Record return").click().run()   # the form's first choice is the worst loan

    assert not at.exception
    after = db.connect(at.db)
    try:
        open_now = circulation.open_loans(after)
    finally:
        after.close()
    assert len(open_now) == before - 1
    assert int(worst["loan_id"]) not in set(open_now["loan_id"])


# ------------------------------------------------------------------ Agent screen
def test_agent_screen_preview_then_run_writes_a_log(app):
    at = app(role="Librarian", section="Agent")
    at.button(key="agent_preview").click().run()
    assert _count(at, "agent_log") == 0 and _count(at, "notifications") == 0

    at.button(key="agent_run").click().run()
    assert not at.exception
    assert _count(at, "agent_log") > 0 and _count(at, "notifications") > 0


def test_agent_screen_run_twice_sends_nothing_twice(app):
    at = app(role="Librarian", section="Agent")
    at.button(key="agent_run").click().run()
    sent, logged = _count(at, "notifications"), _count(at, "agent_log")

    at.button(key="agent_run").click().run()

    assert (_count(at, "notifications"), _count(at, "agent_log")) == (sent, logged)


def test_agent_screen_log_and_messages_show_the_weekday(app):
    at = app(role="Librarian", section="Agent")
    at.button(key="agent_run").click().run()

    log = next(f.value for f in at.dataframe if "run_on" in f.value.columns)
    sent = next(f.value for f in at.dataframe if "created_on" in f.value.columns)
    assert log["run_on"].iloc[0] == dates.short(AS_OF)
    assert sent["created_on"].iloc[0] == dates.short(AS_OF)


def test_agent_screen_advance_one_day(app):
    at = app(role="Librarian", section="Agent")
    at.button(key="agent_advance").click().run()

    assert _library_date(at) == AS_OF + timedelta(days=1)


# ------------------------------------------------------------------ Demand & buying
def test_demand_screen_summarises_the_buy_list(app, conn, forecaster):
    demand = forecaster.book_demand(availability=db.availability_df(conn))
    to_buy = demand[demand["advice"].str.startswith("Buy")]
    page = _page(app(role="Librarian", section="Demand & buying"))

    assert "Titles that need more copies" in page
    assert f"<b>{len(to_buy)}</b>" in page


def test_agent_summary_names_the_day_in_words(app):
    at = app(role="Librarian", section="Agent")
    at.button(key="agent_preview").click().run()

    assert f"for {dates.full(AS_OF)}" in _headings(at)


def test_agent_screen_shows_the_plan_as_readable_tables(app):
    at = app(role="Librarian", section="Agent")
    at.button(key="agent_preview").click().run()
    page = _page(at)

    assert "<th>Student</th>" in page and "Days late" in page
    assert "student_name" not in page                                    # no raw column names on screen