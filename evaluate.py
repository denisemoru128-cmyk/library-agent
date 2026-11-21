"""Measure every AI part of the library agent and write the results for the report.

    python evaluate.py            # full run (about 90 seconds)
    python evaluate.py --quick    # 2 forecast backtests instead of 6 (about 40 seconds)

It writes two files into results/:
    evaluation.md    the tables, ready to paste into the report
    evaluation.json  the same numbers, shown in the app (Librarian > Model quality)

Run it after `python seed.py`. The numbers are repeatable: the data and the models use fixed seeds.
"""
import argparse
import time
from datetime import datetime

from library_agent import db, report
from library_agent.agent import LibraryAgent
from library_agent.chatbot import LibraryChatbot
from library_agent.config import CHATBOT_MIN_CONFIDENCE, DB_PATH, RANDOM_SEED
from library_agent.evaluation import (
    CHATBOT_DEV, CHATBOT_TEST, evaluate_chatbot, evaluate_forecast, evaluate_recommender, evaluate_search,
)
from library_agent.forecast import DemandForecaster
from library_agent.recommender import Recommender
from library_agent.search import BookSearch


def _step(text: str) -> float:
    print(f"  {text} ...", flush=True)
    return time.time()


def collect(quick: bool = False, database=DB_PATH) -> dict:
    """Run every evaluation and return the numbers as plain Python data."""
    conn = db.connect(database)
    try:
        as_of = db.today(conn)
        books, students, loans = db.books_df(conn), db.students_df(conn), db.loans_df(conn)

        started = _step("search")
        search = evaluate_search(books)
        print(f"    done in {time.time() - started:.0f}s")

        started = _step("recommender (all students, then new students with only 3 loans of history)")
        recommender = {
            "established students": evaluate_recommender(books, students, loans, as_of),
            "new students (3 loans)": evaluate_recommender(books, students, loans, as_of, max_history=3),
        }
        print(f"    done in {time.time() - started:.0f}s")

        started = _step("demand forecast backtest (trains one model per test date)")
        forecast = evaluate_forecast(books, loans, as_of, origins=2 if quick else 6)
        print(f"    done in {time.time() - started:.0f}s")

        started = _step("training the final forecaster for the chatbot and the agent")
        forecaster = DemandForecaster(books, loans, as_of)
        print(f"    done in {time.time() - started:.0f}s")

        started = _step("chatbot and agent")
        bot = LibraryChatbot(books, BookSearch(books), Recommender(books, students, loans, as_of), forecaster)
        chatbot = {"dev": evaluate_chatbot(bot, CHATBOT_DEV), "test": evaluate_chatbot(bot, CHATBOT_TEST)}
        agent = LibraryAgent(conn, forecaster)
        plan = agent.decide(agent.perceive())
        agent_snapshot = {"date": as_of.isoformat(), "lines": agent.preview(plan).lines}
        print(f"    done in {time.time() - started:.0f}s")
    finally:
        conn.close()

    return {
        "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M"), "as_of": as_of.isoformat(), "quick": quick,
        "seed": RANDOM_SEED, "books": len(books), "students": len(students), "loans": len(loans),
        "search": search, "recommender": recommender, "forecast": forecast,
        "chatbot": chatbot, "agent": agent_snapshot, "chatbot_min_confidence": CHATBOT_MIN_CONFIDENCE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the library agent's AI components")
    parser.add_argument("--quick", action="store_true", help="2 forecast backtests instead of 6")
    args = parser.parse_args()
    if not DB_PATH.exists():
        raise SystemExit("The database is missing. Run `python seed.py` first.")

    print("Evaluating the library agent")
    results = collect(quick=args.quick)
    markdown_path, json_path = report.save_results(results)
    print("\n" + report.render_markdown(results))
    print(f"Saved to {markdown_path} and {json_path}")


if __name__ == "__main__":
    main()