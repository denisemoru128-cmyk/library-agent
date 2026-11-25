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

From this directory, create and activate a virtual environment, then install the project dependencies
(Streamlit 1.50 or newer is needed):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Generate the simulated SQLite database, run the test suite and start the app:

```powershell
python seed.py
python -m pytest
streamlit run app.py
```

`python seed.py` makes the library's date today's real date, and every weekday on screen is worked out from it.
The saved numbers in `results/` were made with `python seed.py --as-of 2026-11-25`; use that command if you want
to reproduce them exactly, because a different date gives a slightly different simulated history.

The library date is stored in the database, so it falls behind the real date as days pass. The librarian can
bring it up to date with **Set date to today** (in the agent panel) or run `python seed.py` again.
If you run `python seed.py` while the app is open, press **Retrain models** in the sidebar (or restart the app).

## Using the app

The app opens on a sign-in screen. It is a demo sign-in (a registration number such as `REG-0035` is enough, no
password). Three demo buttons are ready: a student in good standing, a student with an overdue book, and the
librarian. **Switch user** is in the menu at the top right, so you can change person without restarting.

- **Student:** Home, Search (type in the box at the top, try `machin lerning`), My loans (Renew and Return; a
  button that is switched off says why underneath), For you. The assistant is docked on the right of every
  screen: try "What is due soon?", or "find books about statistics" and then "is it available?". Each message shows
  its time and two ticks once the bot understood it, and a small line says what it understood and how sure it was.
- **Librarian:** Dashboard (headline numbers, loans per week with the forecast, the books to buy first, and the
  agent's plan on the right), Circulation desk (issue, return, who is overdue), Agent, Demand & buying,
  Catalogue, Model quality.

**Agent demo:** sign in as the librarian. On the Dashboard the panel on the right shows what the agent has
decided. Press *Preview* (changes nothing), then *Run agent* (sends the notices), then *Run agent* again
("Already run today", nothing is sent twice). Press *Advance one day* and run it again to watch it react.

## Measuring the AI parts

```powershell
python evaluate.py                  # one simulated library, about 90 seconds (add --quick for about 40)
python evaluate.py --libraries 5    # five libraries, about 5 minutes: adds the lowest / average / highest table
python -m pytest -v | Out-File -Encoding utf8 results\test_results.txt
```

(`Out-File -Encoding utf8` matters: plain `>` in Windows PowerShell 5 writes UTF-16, which is hard to read in
other tools. `pytest.ini` already stops pytest from treating files in `results/` as tests.)

`evaluate.py` writes `results/evaluation.md` (tables for the report) and `results/evaluation.json` (shown in
the app under Librarian, Model quality). It measures search hit rate, the recommender against popularity and
random baselines on a temporal split, the forecast against naive baselines on a backtest, and the chatbot on
sentences it never trained on. With `--libraries 5` it repeats the recommender and forecast measurements on four
more simulated libraries (each with its own random seed and a different end date) so the report can say how much
the numbers move from one library to the next. The data and the models use fixed seeds, so the numbers repeat.

## How the project is organised

```
app.py            the Streamlit app: loads the models, handles sign-in, picks the screen
seed.py           builds the simulated database
evaluate.py       measures the AI parts and writes results/
library_agent/    the logic (no Streamlit): rules, search, recommender, forecast, chatbot, agent, evaluation
views/            the screens: sign-in, student, librarian, the chat panel, the agent panel, theme
tests/            the test suite (it runs the real app with Streamlit's AppTest)
data/             the catalogue (books.csv) and the generated database
results/          evaluation.md, evaluation.json, test_results.txt
screenshots/      the screens, for the report
assets/           the Karatina University logo and mark
```

`library-agent/` is the project directory; `library_agent/` is the single importable Python package
(underscores are required in package names). The screens only draw what the tested modules decide: for example,
the reason under a switched-off Borrow button comes from the same function that refuses the borrow.

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
  can find such patterns, not how they would score on real students. The scores differ from one simulated
  library to the next; `results/evaluation.md` gives the range.
- The chatbot cannot reliably recognise requests unrelated to the library; it declines only when unsure.
- A book added after the forecaster was trained is searchable at once but joins the demand forecast only after
  **Retrain models**. The same is true after the library date is moved.
- The sign-in is a demo: anyone can pick any account, so it provides no real authentication.