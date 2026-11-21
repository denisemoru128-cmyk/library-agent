# Library Agent

An AI-powered library management agent (Group Practical Assignment 3, project 11). Students search the
catalogue, get recommendations and ask a chatbot. Librarians see demand forecasts with buying advice. An
agent watches the library every day and acts on its own: it reminds students, chases overdue books that
others are waiting for, and proposes which titles to buy.

| Part | AI technique | Where |
|---|---|---|
| Search | TF-IDF word and character n-grams, cosine similarity, typo correction, synonym expansion | `library_agent/search.py` |
| Recommendations | Hybrid of content similarity, item-item collaborative filtering and a cold-start prior, with reasons | `library_agent/recommender.py` |
| Demand forecast | Random forest on weekly loans with calendar features, plus a buy-more-copies rule | `library_agent/forecast.py` |
| Chatbot | TF-IDF + logistic regression intent classifier, entity extraction, FAQ retrieval, conversation memory | `library_agent/chatbot.py` |
| Agent | Perceive, decide, act loop that joins overdue loans with the demand forecast and logs every decision | `library_agent/agent.py` |
| Library rules | Loan period, limit, fines, renewals, reminders (plain rules, not AI) | `library_agent/circulation.py` |

## Set up on Windows

From this directory, create and activate a virtual environment, then install
the project dependencies (Streamlit 1.40 or newer is needed):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Generate the simulated SQLite database, run the test suite and start the app:

```powershell
python seed.py --as-of 2026-11-16
python -m pytest
streamlit run app.py
```

The fixed `--as-of` date makes the demo and every number in `results/` repeatable. Plain `python seed.py` ends
the history on today's date instead, so the numbers would differ slightly from the saved ones.

`library-agent/` is the project directory; `library_agent/` is the single
importable Python package (underscores are required in package names). The
project has one test suite in `tests/`; seeded runtime data is stored in
`data/`. If you run `python seed.py` again while the app is open, press
**Retrain models** in the sidebar (or restart the app).

## Using the app

Choose **Student** or **Librarian** in the sidebar. The library date is simulated (16 November 2026 with the seed command above)
and all people, books and loans are generated.

- **Student:** Search (try `machin lerning`), For you, My loans (renew, return, borrow), Notifications,
  Ask the assistant (try "find books about statistics", then "is it available?").
- **Librarian:** Dashboard, Circulation desk, Agent, Demand & buying, Catalogue, Model quality.

**Agent demo:** Librarian, Agent. Press *Preview* (changes nothing), then *Run agent now*, then run it again
(nothing is sent twice). Press *Advance date by 1 day* and run it again to watch it react.

## Measuring the AI parts

```powershell
python evaluate.py          # about 90 seconds; add --quick for about 40
python -m pytest -v > results\test_results.txt
```

`evaluate.py` writes `results/evaluation.md` (tables for the report) and `results/evaluation.json` (shown in
the app under Librarian, Model quality). It measures search hit rate, the recommender against popularity and
random baselines on a temporal split, the forecast against naive baselines on a backtest, and the chatbot on
sentences it never trained on. The data and the models use a fixed seed, so the numbers repeat.

## Agent and environment

The agent loop is **perceive** (open loans, free copies, the demand forecast), then **reason** (reminders,
overdue and final notices, buy proposals, priority chases), then **act** (send the notices and write every
decision, with its reason, to the agent log). Its environment is the catalog, students, lending rules, and
loan inventory stored in SQLite. Running it twice on the same day sends nothing twice, and a dry run shows
the plan without changing anything.

Library rules (in `library_agent/config.py`): books are lent for 14 days, up to 5 at a time, renewed once, with
a fine of KES 20 per overdue day. A student with an overdue book can not borrow or renew.

## Limits

- The data is simulated, with reading tastes, term and exam seasons built in. The evaluation shows the models
  can find such patterns, not how they would score on real students.
- The chatbot cannot reliably recognise requests unrelated to the library; it declines only when unsure.
- A book added after the forecaster was trained is searchable at once but joins the demand forecast only after
  **Retrain models**.
- Student and librarian views are a demo interface and do not provide account authentication.