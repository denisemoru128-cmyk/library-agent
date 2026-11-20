"""Central settings for the Library Agent.

Library rules and file locations live here so that no other module has
"magic numbers". To change a policy (for example the fine per day), edit it
here once and the whole system follows.
"""
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------- file paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "library.db"
BOOKS_CSV = DATA_DIR / "books.csv"
INTENTS_JSON = DATA_DIR / "intents.json"

# ----------------------------------------------------- library policy (simulated)
LOAN_DAYS = 14               # standard loan period
MAX_ACTIVE_LOANS = 5         # books one student may hold at once
MAX_RENEWALS = 1             # renewals allowed per loan
FINE_PER_DAY = 20            # KES per overdue day, per book
REMINDER_DAYS_BEFORE_DUE = 2 # send a "due soon" reminder this many days before
ESCALATION_DAYS = 14         # overdue longer than this -> final notice + librarian follow-up

# ------------------------------------------------------- academic calendar
# The university year is modelled as repeating blocks: a 15-week term (the last
# 3 weeks are exams) followed by a 2-week break. Borrowing behaviour follows it.
CALENDAR_START = date(2025, 5, 19)   # a Monday: week 1 of a term
TERM_WEEKS = 15
EXAM_WEEKS = 3
BREAK_WEEKS = 2

# ------------------------------------------------------------ simulation
HISTORY_DAYS = 546           # 78 weeks (~18 months) of simulated history
N_STUDENTS = 150
RANDOM_SEED = 42             # same seed -> same dataset, so results are repeatable

# --------------------------------------------------------- AI tuning knobs
SEARCH_MIN_SCORE = 0.08      # below this similarity a search result is dropped
RECOMMEND_TOP_N = 8
FORECAST_HORIZON_WEEKS = 4
CHATBOT_MIN_CONFIDENCE = 0.45  # below this the chatbot admits it is unsure