"""The evaluation pipeline: the numbers it produces, how they are written up, and where they are saved."""
import json

import pytest

from library_agent import report


def test_models_beat_the_simple_baselines(quick_results):
    methods = quick_results["recommender"]["established students"]["methods"]
    forecast = quick_results["forecast"]

    assert methods["hybrid"]["hit_rate"] > methods["popular"]["hit_rate"] > 0
    assert methods["hybrid"]["hit_rate"] > methods["random"]["hit_rate"]
    assert forecast["genre"]["model"]["wape"] < forecast["genre"]["4-week average"]["wape"]
    assert forecast["book"]["model"]["wape"] < forecast["book"]["last 4 weeks repeated"]["wape"]


def test_search_and_chatbot_reach_their_targets(quick_results):
    assert quick_results["search"]["hit_rate"] >= 0.9
    assert quick_results["chatbot"]["test"]["strict_accuracy"] >= 0.8


def test_the_agent_snapshot_describes_its_decisions(quick_results):
    lines = quick_results["agent"]["lines"]

    assert len(lines) == 4 and "overdue" in lines[0]


def test_results_can_be_saved_as_json(quick_results):
    assert json.loads(json.dumps(quick_results))["search"]["queries"] == quick_results["search"]["queries"]


def test_markdown_report_has_every_section_and_the_real_numbers(quick_results):
    text = report.render_markdown(quick_results)
    search = quick_results["search"]

    for heading in ("## 1. Search", "## 2. Recommender", "## 3. Demand forecast", "## 4. Chatbot",
                    "## 5. Agent decisions", "## Limits of these results"):
        assert heading in text
    assert f"{search['hits']} of {search['queries']} queries" in text
    assert "Quick run" in text   # a quick run must say so, so nobody mistakes it for the full six backtests


def test_saved_results_can_be_loaded_again(quick_results, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path / "results")
    assert report.load_results() is None          # nothing saved yet

    markdown_path, json_path = report.save_results(quick_results)

    assert markdown_path.read_text(encoding="utf-8").startswith("# Evaluation results")
    assert report.load_results()["seed"] == quick_results["seed"]