"""The frame around every screen: the sidebar (brand, section links, library date) and the top row
(alerts and the signed-in user).

Section links are ordinary buttons that store the chosen section in ``st.session_state["section"]``.
That keeps the screens easy to test and does not depend on a particular Streamlit version.
"""
import streamlit as st

from library_agent import circulation, db, dates
from library_agent.config import ESCALATION_DAYS

from . import theme

STUDENT_SECTIONS = ["Search", "For you", "My loans", "Notifications", "Assistant"]
LIBRARIAN_SECTIONS = ["Dashboard", "Circulation desk", "Agent", "Demand & buying", "Catalogue", "Model quality"]


def sections_for(user: dict) -> list[str]:
    return STUDENT_SECTIONS if user["role"] == "student" else LIBRARIAN_SECTIONS


def _go(section: str) -> None:
    st.session_state["section"] = section


def sidebar(conn, user: dict, forecaster) -> str:
    """Draw the sidebar and return the section to show."""
    sections = sections_for(user)
    if st.session_state.get("section") not in sections:
        st.session_state["section"] = sections[0]
    current = st.session_state["section"]
    with st.sidebar:
        st.html('<div class="brand">'
                f'<img src="{theme.data_uri("karatina_mark.png")}" alt="">'
                '<div><b>Library Agent</b><small>Karatina University</small></div></div>')
        for name in sections:
            st.button(name, key=f"nav_{name}", type="primary" if name == current else "secondary",
                      on_click=_go, args=(name,))
        st.html('<div class="clock">Library date'
                f'<b>{dates.short(db.today(conn))} {db.today(conn).year}</b>simulated for the demo</div>')
        if user["role"] == "librarian":
            st.html(f'<div class="side-note">Demand model trained on data up to {forecaster.as_of:%d %b %Y}.</div>')
            if st.button("Retrain models", key="retrain"):
                st.session_state["model_version"] = st.session_state.get("model_version", 0) + 1
                st.rerun()
    return current


def alerts(conn, user: dict) -> tuple[int, list[str]]:
    """The number on the Alerts button, and the lines inside it. Worked out from the loans themselves
    (there is no read/unread tracking), so it is always right."""
    lines = []
    if user["role"] == "student":
        loans = circulation.open_loans(conn, user["student_id"]).sort_values("days_left")
        for row in loans[loans["days_left"] < 0].itertuples():
            lines.append(f"**Overdue.** “{row.title}” was due {dates.short(row.due_on.date())} "
                         f"({-row.days_left} days ago). Fine so far: KES {row.fine}.")
        for row in loans[loans["status"] == "due soon"].itertuples():
            lines.append(f"**Due soon.** “{row.title}” is due {dates.short(row.due_on.date())}.")
        count = len(lines)
        for row in db.notifications_df(conn, user["student_id"], limit=3).itertuples():
            lines.append(f"From the library, {row.created_on}: {row.message}")
        return count, lines
    overdue = circulation.overdue_loans(conn)
    serious = overdue[overdue["days_overdue"] > ESCALATION_DAYS].sort_values("days_overdue", ascending=False)
    for row in serious.head(4).itertuples():
        lines.append(f"**{row.student_name}** is {row.days_overdue} days overdue on “{row.title}”.")
    if len(serious) > 4:
        lines.append(f"and {len(serious) - 4} more loans are over {ESCALATION_DAYS} days overdue.")
    return len(serious), lines


def top_row(conn, user: dict, students) -> None:
    """Alerts and the user menu, right-aligned above the page."""
    count, lines = alerts(conn, user)
    if user["role"] == "student":
        student = students[students["student_id"] == user["student_id"]].iloc[0]
        who, detail = f"{student['name']} · {student['reg_no']}", (
            f"{student['department']}  \nInterests: {str(student['interests']).replace('|', ', ')}")
    else:
        who, detail = "Librarian", "Library desk"
    _, alerts_col, user_col = st.columns([6, 1.3, 2.4])
    with alerts_col:
        with st.popover(f"Alerts ({count})" if count else "Alerts"):
            if lines:
                for line in lines:
                    st.markdown(line)
            else:
                st.markdown("Nothing needs your attention.")
    with user_col:
        with st.popover(who):
            st.markdown(detail)
            if st.button("Switch user", key="switch_user"):
                for key in ("user", "section", "_flash"):
                    st.session_state.pop(key, None)
                st.rerun()