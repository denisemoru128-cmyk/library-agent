"""The library agent's plan, docked on the librarian Dashboard, and the buttons that run it.

The panel shows what the agent has decided (final notices, priority chases) before anything is sent.
"Preview" changes nothing. "Run agent" sends the notices and writes every decision to the log; running it
twice on the same day sends nothing twice (the agent itself guarantees that).
The same buttons are used on the Agent screen, with another key prefix.
"""
from datetime import date, timedelta

import streamlit as st

from library_agent import dates, db
from library_agent.agent import LibraryAgent
from library_agent.config import ESCALATION_DAYS

from . import components as ui
from .common import Models, flash, go

SHOWN_NOTICES = 3
SHOWN_CHASES = 2


# ------------------------------------------------------------------ the buttons
def run_buttons(conn, models: Models, prefix: str) -> None:
    """Preview and Run agent. The outcome is kept in the session so the panel and the Agent screen can show it."""
    preview, run = st.columns(2)
    wants_preview = preview.button("Preview", key=f"{prefix}_preview")
    wants_run = run.button("Run agent", key=f"{prefix}_run", type="primary")
    if wants_preview or wants_run:
        agent = LibraryAgent(conn, models.forecaster)
        plan = agent.decide(agent.perceive())
        cycle = agent.act(plan) if wants_run else agent.preview(plan)
        st.session_state["agent_result"] = {"on": plan.on, "executed": cycle.executed,
                                            "new": sum(cycle.notifications.values()), "summary": cycle.summary()}


def _set_clock(conn, day: date, message: str) -> None:
    db.set_today(conn, day)
    st.session_state.pop("agent_result", None)
    flash("info", message)
    st.rerun()


def clock_buttons(conn, prefix: str) -> None:
    """Move the simulated library date: one day on, or back to the real date (it is stored in the database,
    so it goes stale as real days pass)."""
    today, real = db.today(conn), date.today()
    if st.button("Advance one day", key=f"{prefix}_advance", type="tertiary"):
        tomorrow = today + timedelta(days=1)
        _set_clock(conn, tomorrow, f"The library date is now {dates.full(tomorrow)}. Run the agent to see it react.")
    if today != real and st.button(f"Set date to today ({dates.short(real)})", key=f"{prefix}_today", type="tertiary"):
        _set_clock(conn, real, f"The library date is now today, {dates.full(real)}.")


# ------------------------------------------------------------------ the panel
def _banner(result: dict | None, on: date) -> str:
    if result and result["on"] == on and result["executed"]:
        if result["new"]:
            return (f'<div class="plan-note done"><b>Done.</b>The agent sent {ui.plural(result["new"], "new notice")} '
                    "and wrote its decisions to the log.</div>")
        return '<div class="plan-note done"><b>Already run today.</b>Nothing new to send.</div>'
    return '<div class="plan-note"><b>Preview only.</b>Nothing has been sent. Run the agent to send the notices.</div>'


def _group(title: str, count: int, sub: str, items: list[str], none_text: str) -> str:
    body = "".join(items) if items else f'<div class="pl-none">{ui.esc(none_text)}</div>'
    return (f'<div class="pl-h"><b>{ui.esc(title)}</b><span class="count">{count}</span></div>'
            f'<div class="pl-sub">{ui.esc(sub)}</div>{body}')


def _item(name: str, what: str, detail: str, late: bool = False) -> str:
    return (f'<div class="pl-item"><b>{ui.esc(name)}</b><span>{ui.esc(what)}</span>'
            f'<small class="{"late" if late else ""}">{ui.esc(detail)}</small></div>')


def _more(count: int, shown: int, key: str) -> None:
    if count > shown:
        st.button(f"and {count - shown} more", key=f"plan_more_{key}", type="tertiary", on_click=go, args=("Agent",))


def render(conn, models: Models) -> None:
    """Draw the panel on the right. It is a fixed container, so where it is drawn in the script does not matter."""
    with st.container(key="plan_panel"):
        head = st.container(key="plan_head")
        body = st.container(key="plan_body")
        foot = st.container(key="plan_foot")

    with foot:                                         # first, so a click is reflected in the body below
        run_buttons(conn, models, "plan")
        clock_buttons(conn, "plan")

    agent = LibraryAgent(conn, models.forecaster)
    plan = agent.decide(agent.perceive())
    with head:
        st.html(f'<div class="chat-h"><b>Agent</b><small>Plan for {ui.esc(dates.full(plan.on))}</small></div>')
    with body:
        st.html(_banner(st.session_state.get("agent_result"), plan.on))
        notices = [_item(row.student_name, row.title, f"{row.days_overdue} days · KES {row.fine}", late=True)
                   for row in plan.escalations.sort_values("days_overdue", ascending=False).head(SHOWN_NOTICES).itertuples()]
        st.html(_group("Final notices", len(plan.escalations),
                       f"More than {ESCALATION_DAYS} days overdue. The student gets a final notice and you follow up.",
                       notices, f"No loan is more than {ESCALATION_DAYS} days overdue."))
        _more(len(plan.escalations), SHOWN_NOTICES, "notices")
        chases = [_item(row.student_name, row.title, f"{row.days_overdue} days late · {row.forecast_loans:.1f} loans forecast")
                  for row in plan.chases.sort_values("forecast_loans", ascending=False).head(SHOWN_CHASES).itertuples()]
        st.html(_group("Chase first", len(plan.chases), "Late, every copy is out, and demand is forecast.",
                       chases, "No overdue copy is holding up a book that is in demand."))
        _more(len(plan.chases), SHOWN_CHASES, "chases")
        st.html(f'<div class="pl-sub" style="margin-top:14px">Also today: {ui.plural(len(plan.due_soon), "loan")} due soon '
                f'get a reminder and {ui.plural(len(plan.overdue), "overdue loan")} get an overdue notice.</div>')