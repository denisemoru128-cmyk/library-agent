"""The multi-library evaluation: the settings of each library, the lowest/average/highest summary and the report text.

These use small hand-made results (no model is trained), so they run in a fraction of a second.
"""
import copy
from datetime import date

from library_agent import report, stability


def fake_run(hybrid=0.30, popular=0.20, content=0.25, genre=0.10, genre_base=0.15, search=0.95):
    methods = {"hybrid": {"hit_rate": hybrid, "recall": 0.1}, "content": {"hit_rate": content, "recall": 0.1},
               "collaborative": {"hit_rate": 0.1, "recall": 0.1}, "popular": {"hit_rate": popular, "recall": 0.1},
               "random": {"hit_rate": 0.02, "recall": 0.01}}
    return {
        "search": {"hit_rate": search},
        "recommender": {"established students": {"methods": methods}, "new students (3 loans)": {"methods": methods}},
        "forecast": {"genre": {"model": {"wape": genre}, "4-week average": {"wape": genre_base}},
                     "book": {"model": {"wape": 0.4}, "last 4 weeks repeated": {"wape": 0.5}}},
        "chatbot": {"test": {"strict_accuracy": 0.9}},
    }


SETTINGS = stability.library_settings(42, date(2026, 11, 25), 3)


def test_each_library_has_its_own_seed_and_end_date():
    assert SETTINGS == [(42, date(2026, 11, 25)), (43, date(2026, 11, 14)), (44, date(2026, 11, 3))]


def test_summary_gives_lowest_average_and_highest():
    runs = [fake_run(hybrid=0.20), fake_run(hybrid=0.30), fake_run(hybrid=0.40)]
    metric = stability.summarise(runs, SETTINGS)["metrics"]["rec_hybrid"]

    assert (metric["min"], metric["max"]) == (0.20, 0.40)
    assert abs(metric["mean"] - 0.30) < 1e-9
    assert metric["values"] == [0.20, 0.30, 0.40]


def test_comparisons_count_how_often_the_model_wins_and_ties():
    runs = [fake_run(hybrid=0.30, content=0.25),     # hybrid ahead
            fake_run(hybrid=0.25, content=0.25),     # tie
            fake_run(hybrid=0.20, content=0.25)]     # hybrid behind
    found = {c["text"]: c for c in stability.summarise(runs, SETTINGS)["comparisons"]}
    versus_content = found["Hybrid recommender beats content-based only (established students)"]

    assert (versus_content["ahead"], versus_content["ties"], versus_content["of"]) == (1, 1, 3)


def test_lower_is_better_for_forecast_error():
    runs = [fake_run(genre=0.10, genre_base=0.15), fake_run(genre=0.20, genre_base=0.15)]
    found = {c["text"]: c for c in stability.summarise(runs, SETTINGS[:2])["comparisons"]}

    assert found["Forecast beats the 4-week average (by genre)"]["ahead"] == 1


def test_evaluate_many_builds_the_other_libraries_and_keeps_the_first_result():
    built, collected = [], []

    def make_library(path, as_of, seed):
        built.append((seed, as_of))

    def collect(quick, database):
        collected.append(quick)
        return fake_run(hybrid=0.10 * len(collected))

    first = fake_run(hybrid=0.99)
    summary = stability.evaluate_many(first, 3, collect, date(2026, 11, 25), True, make_library=make_library)

    assert built == [(43, date(2026, 11, 14)), (44, date(2026, 11, 3))]       # the demo library is not rebuilt
    assert collected == [True, True]
    assert summary["metrics"]["rec_hybrid"]["values"] == [0.99, 0.1, 0.2]
    assert [lib["seed"] for lib in summary["libraries"]] == [42, 43, 44]


def test_range_text_is_empty_for_a_single_library():
    assert stability.range_text(None, "rec_hybrid") == ""
    summary = stability.summarise([fake_run(hybrid=0.2), fake_run(hybrid=0.4)], SETTINGS[:2])
    assert stability.range_text(summary, "rec_hybrid") == "20% to 40%"
    assert stability.range_text(summary, "genre_model") == "10.0% to 10.0%"


# ---- the report -------------------------------------------------------------------------------------
def test_report_gets_a_stability_section_only_when_several_libraries_were_run(quick_results):
    assert "## 6. How stable are these numbers?" not in report.render_markdown(quick_results)

    several = copy.deepcopy(quick_results)
    several["stability"] = stability.summarise([fake_run(), fake_run(hybrid=0.35)], SETTINGS[:2])
    text = report.render_markdown(several)

    assert "## 6. How stable are these numbers? (2 simulated libraries)" in text
    assert "| Measure | Lowest | Average | Highest |" in text
    assert "**2 of 2 libraries**" in text
    assert text.index("## 6.") < text.index("## Limits of these results")