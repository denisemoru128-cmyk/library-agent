"""Turn the evaluation numbers into text for the report, and save or load them.

evaluate.py produces the numbers (a plain dictionary). This module writes them as Markdown
tables with a plain-language note under each, so the same words appear in the report file and in
the app's "Model quality" section.
"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

METHOD_NAMES = {
    "hybrid": "Hybrid (our model)", "content": "Content-based only", "collaborative": "Collaborative only",
    "popular": "Most popular (baseline)", "random": "Random (baseline)",
}


def save_results(results: dict) -> tuple[Path, Path]:
    """Write evaluation.md and evaluation.json into results/. Returns both paths."""
    RESULTS_DIR.mkdir(exist_ok=True)
    markdown_path, json_path = RESULTS_DIR / "evaluation.md", RESULTS_DIR / "evaluation.json"
    markdown_path.write_text(render_markdown(results), encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return markdown_path, json_path


def load_results() -> dict | None:
    """The last saved results, or None if evaluate.py has not been run yet."""
    path = RESULTS_DIR / "evaluation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(r: dict) -> str:
    """The results as Markdown, with a plain-language note under each table."""
    out = [
        "# Evaluation results", "",
        f"Generated {r['generated_on']} on simulated data ({r['books']} books, {r['students']} students, "
        f"{r['loans']:,} loans, library date {r['as_of']}, seed {r['seed']}).",
    ]
    if r["quick"]:
        out += ["", "**Quick run:** only 2 forecast backtests were used. Run `python evaluate.py` for the full 6."]

    s = r["search"]
    out += ["", "## 1. Search", "",
            f"Hit rate at {s['k']}: a correct book is in the top {s['k']} results for "
            f"**{s['hits']} of {s['queries']} queries ({s['hit_rate']:.0%})**. The queries include misspellings, "
            "descriptions with no title words, and author names."]
    if s["misses"]:
        out += ["", "Missed: " + "; ".join(f"“{q}” (returned: {', '.join(top)})" for q, top in s["misses"]) + "."]

    out += ["", "## 2. Recommender", "",
            "Temporal split: the model sees loans only up to 60 days ago, then we check whether the books each student "
            "really borrowed afterwards are in their top 10. Hit rate = students with at least one correct book; "
            "recall = share of their future books found."]
    for group, result in r["recommender"].items():
        out += ["", f"**{group}** ({result['students_tested']} students tested)", "",
                _table(["Method", "Hit rate @10", "Recall @10"],
                       [[METHOD_NAMES.get(m, m), f"{v['hit_rate']:.0%}", f"{v['recall']:.0%}"]
                        for m, v in result["methods"].items()])]
    cold = r["recommender"].get("new students (3 loans)", {}).get("methods", {})
    if cold and cold["popular"]["hit_rate"] < cold["random"]["hit_rate"]:
        out += ["", "“Most popular” scores below random for new students. A likely reason is that in this simulated "
                    "library students borrow mostly within their own interests, which one list for everyone ignores."]

    f = r["forecast"]
    out += ["", "## 3. Demand forecast", "",
            f"Backtest: {f['origins']} past dates, each forecasting the next {f['horizon']} weeks using only earlier data. "
            "WAPE is total error divided by total real loans (lower is better). Baselines need no AI: repeat the "
            "recent average.", "",
            _table(["Level", "Method", "Average error (loans)", "WAPE"],
                   [[level, name, f"{v['mae']:.2f}", f"{v['wape']:.1%}"]
                    for level in ("genre", "book") for name, v in f[level].items()]),
            "", "Per-book forecasts are much noisier than per-genre ones, because one book has few loans per week."]

    c = r["chatbot"]
    out += ["", "## 4. Chatbot (intent recognition on sentences it never trained on)", "",
            f"Strict accuracy counts a sentence as right only if the intent is correct AND the bot was at least "
            f"{r['chatbot_min_confidence']:.0%} sure. “Accuracy when sure” ignores the sentences it declined to answer.", "",
            _table(["Set", "Sentences", "Strict accuracy", "Accuracy when sure", "Answered"],
                   [[name.upper() + (" (used while tuning)" if name == "dev" else " (held out)"), v["sentences"],
                     f"{v['strict_accuracy']:.0%}", f"{v['accuracy_if_sure']:.0%}", f"{v['answered']:.0%}"]
                    for name, v in c.items()]), "",
            "Sentences the bot got wrong on the held-out set (expected intent, what it said, confidence):", ""]
    out += [f"- “{q}”: expected **{want}**, got *{got}* ({conf:.0%})" for q, want, got, conf in c["test"]["wrong"]]
    if any(q == "can i renew a loan" for q, *_ in c["test"]["wrong"]):
        out += ["", "“can i renew a loan” counts as wrong because the classifier alone says *renew*. The chat handler "
                    "turns questions like this into the rules answer, so a student still gets the right reply "
                    "and nothing is renewed."]

    a = r["agent"]
    out += ["", f"## 5. Agent decisions on {a['date']} (dry run, nothing changed)", ""]
    out += [f"- {line}" for line in a["lines"]]

    out += ["", "## Limits of these results", "",
            "- All data is **simulated**. The simulator builds in reading tastes, seasons and exam rushes, so these "
            "numbers show that each model can find such patterns, not how it would score on real students.",
            "- The chatbot's two sentence sets are small. The TEST set was used to find two bugs after its first "
            "run, so its score is slightly optimistic.",
            "- The chatbot can not reliably recognise requests that have nothing to do with the library; it only "
            "declines when it is unsure.", ""]
    return "\n".join(out)