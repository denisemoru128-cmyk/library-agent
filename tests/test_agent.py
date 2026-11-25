from datetime import timedelta

import pytest

from library_agent import circulation, db
from library_agent.agent import LibraryAgent
from library_agent.config import ESCALATION_DAYS
from tests.conftest import AS_OF


@pytest.fixture()
def agent(fresh_conn, forecaster):
    return LibraryAgent(fresh_conn, forecaster)


def _counts(conn):
    return (len(db.notifications_df(conn)), len(db.agent_log_df(conn)))


def test_dry_run_changes_nothing(agent):
    before = _counts(agent.conn)
    report = agent.run_cycle(dry_run=True)
    assert not report.executed and report.lines
    assert _counts(agent.conn) == before


def test_cycle_notifies_every_overdue_and_due_soon_loan(agent):
    report = agent.run_cycle()
    assert report.executed
    assert report.notifications["overdue"] == len(circulation.overdue_loans(agent.conn))
    assert report.notifications["due_soon"] == len(circulation.due_soon(agent.conn))


def test_final_notice_only_for_long_overdue_loans(agent):
    plan = agent.decide(agent.perceive())
    assert (plan.escalations["days_overdue"] > ESCALATION_DAYS).all()
    assert len(plan.escalations) == (plan.overdue["days_overdue"] > ESCALATION_DAYS).sum()
    report = agent.act(plan)
    assert report.notifications["final_notice"] == len(plan.escalations)


def test_acquisition_proposals_match_the_forecast_advice(agent):
    plan = agent.decide(agent.perceive())
    assert not plan.acquisitions.empty
    assert plan.acquisitions["advice"].str.startswith("Buy").all()
    agent.act(plan)
    log = db.agent_log_df(agent.conn)
    assert (log["action"] == "acquisition_proposal").sum() == len(plan.acquisitions)


def test_priority_chase_combines_overdue_with_demand(agent):
    plan = agent.decide(agent.perceive())
    assert not plan.chases.empty
    # every chase is an overdue loan of a book with no free copy and real demand
    assert (plan.chases["days_overdue"] > 0).all()
    demand = agent.forecaster.book_demand(availability=db.availability_df(agent.conn)).set_index("book_id")
    assert (demand.loc[plan.chases["book_id"], "available"] == 0).all()
    assert (demand.loc[plan.chases["book_id"], "advice"] != "OK").all()


def test_running_twice_on_the_same_day_adds_nothing(agent):
    first = agent.run_cycle()
    after_first = _counts(agent.conn)
    second = agent.run_cycle()
    assert first.log_entries > 0
    assert second.log_entries == 0 and sum(second.notifications.values()) == 0
    assert _counts(agent.conn) == after_first


def test_agent_notices_when_the_world_changes(agent):
    """Perceive-reason-act: after a late book is returned, the agent stops chasing it."""
    before = agent.decide(agent.perceive()).overdue
    returned = before.iloc[0]
    assert circulation.return_book(agent.conn, int(returned["loan_id"])).ok
    after = agent.decide(agent.perceive()).overdue
    assert int(returned["loan_id"]) not in set(after["loan_id"])
    assert len(after) == len(before) - 1


def test_next_day_sends_fresh_reminders_with_larger_fines(agent):
    agent.run_cycle()
    tomorrow = AS_OF + timedelta(days=1)
    report = agent.run_cycle(on=tomorrow)
    assert report.notifications["overdue"] > 0            # a new day, a new overdue notice
    latest = db.notifications_df(agent.conn, limit=1).iloc[0]
    assert latest["created_on"] == tomorrow.isoformat()