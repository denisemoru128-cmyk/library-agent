"""How stable are the evaluation numbers? Run the whole evaluation on several simulated libraries and summarise.

One simulated library gives one number per model, and one lucky (or unlucky) library could mislead. So
evaluate.py can build several libraries (each with its own random seed, so different students, tastes and
loans, and a history that ends on a different day) and run the same evaluation on each. This module turns the
list of results into a lowest / average / highest table and counts how often a model beats its baseline.

Nothing here trains a model. It only reads the dictionaries that evaluate.collect() returns.
"""
import tempfile
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

from .config import RANDOM_SEED
from .simulation import seed_database

DAYS_BETWEEN_END_DATES = 11          # library 2 ends 11 days earlier than library 1, library 3 22 days earlier, ...

# id -> (label, how to read it from one evaluation result, digits when printed as a percentage).
# Search and the chatbot are left out on purpose: they are tested on the catalogue and on fixed sentence sets,
# not on loans, so they would give the same number for every library and say nothing about stability.
METRICS = {
    "rec_hybrid": ("Recommender hit rate @10: hybrid (established students)",
                   lambda r: r["recommender"]["established students"]["methods"]["hybrid"]["hit_rate"], 0),
    "rec_content": ("Recommender hit rate @10: content-based only",
                    lambda r: r["recommender"]["established students"]["methods"]["content"]["hit_rate"], 0),
    "rec_popular": ("Recommender hit rate @10: most popular",
                    lambda r: r["recommender"]["established students"]["methods"]["popular"]["hit_rate"], 0),
    "rec_new_hybrid": ("Recommender hit rate @10: hybrid (new students)",
                       lambda r: r["recommender"]["new students (3 loans)"]["methods"]["hybrid"]["hit_rate"], 0),
    "rec_new_popular": ("Recommender hit rate @10: most popular (new students)",
                        lambda r: r["recommender"]["new students (3 loans)"]["methods"]["popular"]["hit_rate"], 0),
    "genre_model": ("Forecast error by genre (WAPE): model", lambda r: r["forecast"]["genre"]["model"]["wape"], 1),
    "genre_base": ("Forecast error by genre (WAPE): 4-week average",
                   lambda r: r["forecast"]["genre"]["4-week average"]["wape"], 1),
    "book_model": ("Forecast error by book (WAPE): model", lambda r: r["forecast"]["book"]["model"]["wape"], 1),
    "book_base": ("Forecast error by book (WAPE): last 4 weeks repeated",
                  lambda r: r["forecast"]["book"]["last 4 weeks repeated"]["wape"], 1),
}

# (sentence, id that should be better, id it is compared with, True if a higher number is better)
COMPARISONS = [
    ("Hybrid recommender beats “most popular” (established students)", "rec_hybrid", "rec_popular", True),
    ("Hybrid recommender beats content-based only (established students)", "rec_hybrid", "rec_content", True),
    ("Hybrid recommender beats “most popular” (new students)", "rec_new_hybrid", "rec_new_popular", True),
    ("Forecast beats the 4-week average (by genre)", "genre_model", "genre_base", False),
    ("Forecast beats repeating the last 4 weeks (by book)", "book_model", "book_base", False),
]


def library_settings(first_seed: int, first_end: date, count: int) -> list[tuple[int, date]]:
    """(seed, last day of history) for each library. The first is the demo library itself."""
    return [(first_seed + i, first_end - timedelta(days=DAYS_BETWEEN_END_DATES * i)) for i in range(count)]


def summarise(runs: list[dict], settings: list[tuple[int, date]]) -> dict:
    """Lowest, average and highest of every metric, and how often each model beat its baseline."""
    metrics = {}
    for key, (label, read, digits) in METRICS.items():
        values = [read(run) for run in runs]
        metrics[key] = {"label": label, "digits": digits, "values": values,
                        "min": min(values), "mean": mean(values), "max": max(values)}
    comparisons = []
    for text, better, other, higher_is_better in COMPARISONS:
        a, b = metrics[better]["values"], metrics[other]["values"]
        ahead = sum((x > y) if higher_is_better else (x < y) for x, y in zip(a, b))
        ties = sum(x == y for x, y in zip(a, b))
        comparisons.append({"text": text, "ahead": ahead, "ties": ties, "of": len(runs)})
    return {"libraries": [{"seed": seed, "as_of": day.isoformat()} for seed, day in settings],
            "metrics": metrics, "comparisons": comparisons}


def evaluate_many(first: dict, count: int, collect, as_of: date, quick: bool,
                  make_library=seed_database, first_seed: int = RANDOM_SEED) -> dict:
    """``first`` is the demo library's results (already computed). The other libraries are built in a
    temporary folder, evaluated with ``collect`` and thrown away. Returns summarise()'s dictionary."""
    settings = library_settings(first_seed, as_of, count)
    runs = [first]
    with tempfile.TemporaryDirectory() as folder:
        for number, (seed, end) in enumerate(settings[1:], start=2):
            path = Path(folder) / f"library_{number}.db"
            print(f"\nLibrary {number} of {count} (seed {seed}, history ends {end})", flush=True)
            make_library(path, as_of=end, seed=seed)
            runs.append(collect(quick=quick, database=path))
    return summarise(runs, settings)


def show(value: float, digits: int) -> str:
    return f"{value:.{digits}%}"


def range_text(stability: dict | None, key: str) -> str:
    """'18% to 25%' for one metric, or '' when the evaluation was run on a single library."""
    if not stability:
        return ""
    metric = stability["metrics"][key]
    return f"{show(metric['min'], metric['digits'])} to {show(metric['max'], metric['digits'])}"