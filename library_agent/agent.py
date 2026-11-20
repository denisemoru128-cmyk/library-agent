"""The library agent: perceive -> reason -> act, on its own.

This is the "intelligent agent" of the project. One call to ``run_cycle()`` is
one working day of a librarian who never forgets:

PERCEIVE  Look at the environment: every loan still out, which copies are free,
          and the demand forecast for the coming weeks.

REASON    Combine what it sees into decisions:
            * a loan due within 2 days            -> remind the student
            * a loan past its due date            -> overdue notice (with the fine so far)
            * overdue by more than 14 days        -> FINAL notice + flag for the librarian
            * forecast says we are short of copies -> propose buying more
            * an overdue copy of a book that has NO free copy and is in demand
                                                  -> priority chase (everyone is waiting for it)
          The last rule joins two separate sources (overdue loans and the demand
          forecast), which is what makes this reasoning rather than a checklist.

ACT       Send the notifications, and write every decision, with its reason, to
          the agent log, so a human can see WHAT the agent did and WHY.

Safe to run repeatedly: notifications and log entries are de-duplicated per day.
``dry_run=True`` shows what the agent *would* do without changing anything.
"""
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from . import circulation, db
from .config import ESCALATION_DAYS
from .forecast import DemandForecaster


@dataclass
class State:
    """What the agent perceives."""
    on: date
    open_loans: pd.DataFrame
    demand: pd.DataFrame          # one row per book: copies, available, forecast, advice


@dataclass
class Plan:
    """What the agent has decided to do (before doing it)."""
    on: date
    due_soon: pd.DataFrame
    overdue: pd.DataFrame
    escalations: pd.DataFrame
    acquisitions: pd.DataFrame
    chases: pd.DataFrame


@dataclass
class CycleReport:
    on: date
    executed: bool
    notifications: dict = field(default_factory=dict)   # kind -> number of NEW messages
    log_entries: int = 0                                 # NEW diary lines
    lines: list = field(default_factory=list)            # readable summary of the decisions

    def summary(self) -> str:
        head = "Agent cycle" if self.executed else "Agent plan (dry run, nothing changed)"
        return f"{head} for {self.on}:\n" + "\n".join(f"- {line}" for line in self.lines)


class LibraryAgent:
    def __init__(self, conn, forecaster: DemandForecaster | None = None):
        self.conn = conn
        # Training the forecaster takes a few seconds, so callers may pass one in to reuse it.
        self.forecaster = forecaster or DemandForecaster(db.books_df(conn), db.loans_df(conn), db.today(conn))

    # ------------------------------------------------------------ 1. perceive
    def perceive(self, on: date | None = None) -> State:
        on = on or db.today(self.conn)
        return State(
            on=on,
            open_loans=circulation.open_loans(self.conn, on=on),
            demand=self.forecaster.book_demand(availability=db.availability_df(self.conn)),
        )

    # ------------------------------------------------------------- 2. reason
    def decide(self, state: State) -> Plan:
        loans = state.open_loans
        overdue = loans[loans["days_left"] < 0].copy()
        overdue["days_overdue"] = -overdue["days_left"]
        overdue = overdue.sort_values("days_overdue", ascending=False).reset_index(drop=True)
        due_soon = loans[loans["status"] == "due soon"].reset_index(drop=True)
        escalations = overdue[overdue["days_overdue"] > ESCALATION_DAYS].reset_index(drop=True)

        acquisitions = state.demand[state.demand["advice"].str.startswith("Buy")].reset_index(drop=True)

        # Join two sources of knowledge: who is late, and which books people are waiting for.
        pressured = state.demand[(state.demand["available"] == 0) & (state.demand["advice"] != "OK")]
        chases = overdue.merge(
            pressured[["book_id", "forecast_loans", "advice"]], on="book_id", how="inner"
        ).sort_values(["forecast_loans", "days_overdue"], ascending=False).reset_index(drop=True)

        return Plan(state.on, due_soon, overdue, escalations, acquisitions, chases)

    # --------------------------------------------------------------- 3. act
    def act(self, plan: Plan) -> CycleReport:
        on, conn = plan.on, self.conn
        report = CycleReport(on=on, executed=True)

        # Reminders and overdue notices come from circulation, which already de-duplicates by day.
        report.notifications = dict(circulation.create_reminders(conn, on))
        report.notifications["final_notice"] = 0

        for row in plan.escalations.itertuples():
            message = (f"FINAL NOTICE: “{row.title}” is {row.days_overdue} days overdue "
                       f"(fine so far KES {row.fine}). Borrowing is suspended until it is returned.")
            report.notifications["final_notice"] += db.insert_notification(
                conn, row.student_id, row.loan_id, "final_notice", message, on)
            report.log_entries += db.log_action(
                conn, on, "escalation",
                f"{row.student_name} ({row.reg_no}) is {row.days_overdue} days overdue on "
                f"“{row.title}” (more than {ESCALATION_DAYS}). Final notice sent; "
                f"librarian should follow up. Fine so far: KES {row.fine}.")

        for row in plan.acquisitions.itertuples():
            gap = row.copies_needed - row.copies
            report.log_entries += db.log_action(
                conn, on, "acquisition_proposal",
                f"Buy {gap} more cop{'y' if gap == 1 else 'ies'} of “{row.title}”: "
                f"{row.copies} owned, {row.forecast_loans} loans forecast over 4 weeks "
                f"(needs {row.copies_needed} copies to stay under 70% busy).")

        for row in plan.chases.itertuples():
            report.log_entries += db.log_action(
                conn, on, "priority_chase",
                f"Chase {row.student_name} ({row.reg_no}) for “{row.title}”, "
                f"{row.days_overdue} days overdue: no copy is free and {row.forecast_loans} loans are forecast.")

        sent = {kind: n for kind, n in report.notifications.items() if n}
        if sent:  # only log when something NEW was sent, so a repeat run adds nothing
            report.log_entries += db.log_action(
                conn, on, "notifications_sent", ", ".join(f"{n} {kind.replace('_', ' ')}" for kind, n in sent.items()))
        report.lines = self._describe(plan, report)
        report.log_entries += db.log_action(conn, on, "cycle_summary", "; ".join(self._describe(plan)))
        conn.commit()
        return report

    # ---------------------------------------------------------------- driver
    def run_cycle(self, on: date | None = None, dry_run: bool = False) -> CycleReport:
        """One full perceive -> reason -> act pass. ``dry_run`` stops before acting."""
        plan = self.decide(self.perceive(on))
        if dry_run:
            return CycleReport(on=plan.on, executed=False, lines=self._describe(plan))
        return self.act(plan)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _describe(plan: Plan, report: CycleReport | None = None) -> list:
        lines = [
            f"{len(plan.overdue)} overdue loans, {len(plan.due_soon)} due soon",
            f"{len(plan.escalations)} overdue by more than {ESCALATION_DAYS} days (final notice)",
            f"{len(plan.acquisitions)} books where the forecast says we need more copies",
            f"{len(plan.chases)} overdue copies of books that are fully out and in demand (priority chase)",
        ]
        if report is not None:
            sent = {k: v for k, v in report.notifications.items() if v}
            lines.append("new notifications: " + (", ".join(f"{v} {k}" for k, v in sent.items()) or "none (already sent today)"))
        return lines