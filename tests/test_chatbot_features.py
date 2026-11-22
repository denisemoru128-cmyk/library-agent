"""Tests for the chatbot's conversation features: memory, rules, loans, renewals, honesty.

The five basic tests in test_chatbot.py still apply; these cover what was added on top.
"""
from datetime import date, timedelta

import pytest

from library_agent import circulation, dates, db
from library_agent.chat_data import INTENT_EXAMPLES
from library_agent.chatbot import LibraryChatbot
from library_agent.evaluation import CHATBOT_DEV, CHATBOT_TEST, evaluate_chatbot
from library_agent.recommender import Recommender
from library_agent.search import BookSearch
from tests.conftest import AS_OF, student_with_late_and_ok_loan


@pytest.fixture(scope="module")
def bot(conn, forecaster):
    books, students, loans = db.books_df(conn), db.students_df(conn), db.loans_df(conn)
    return LibraryChatbot(books, BookSearch(books), Recommender(books, students, loans, AS_OF), forecaster)


def _clean_loan(conn):
    """A student who is not overdue and still has a renewal left, with a plain title to quote."""
    loans = circulation.open_loans(conn)
    late = set(loans.loc[loans["days_left"] < 0, "student_id"])
    ok = loans[~loans["student_id"].isin(late) & (loans["renewals"] == 0)
               & ~loans["title"].str.contains("['\",]")]
    row = ok.iloc[0]
    return int(row["student_id"]), int(row["loan_id"]), row["title"]


def _overdue_student(conn):
    loans = circulation.open_loans(conn)
    return int(loans[loans["days_left"] < 0].iloc[0]["student_id"])


# ---- library rules (FAQ) ------------------------------------------------------
def test_policy_answers_come_from_the_real_rules(bot, conn):
    loan_period = bot.answer("how long can i keep a book", conn)
    fines = bot.answer("what is the fine for late books", conn)

    assert loan_period["intent"] == "policy" and "14 days" in loan_period["message"]
    assert "KES 20" in fines["message"]


# ---- memory of the conversation -----------------------------------------------
def test_follow_up_uses_the_book_from_the_previous_message(bot, conn):
    context = {}
    first = bot.answer("find books about statistics", conn, context=context)
    follow_up = bot.answer("is it available?", conn, context=context)

    assert first["intent"] == "search" and context["last_book"] == first["data"][0]["title"]
    assert follow_up["intent"] == "availability"
    assert follow_up["message"].startswith("Assuming you mean")
    assert follow_up["data"][0]["title"] == first["data"][0]["title"]


def test_without_context_a_vague_follow_up_asks_for_a_title(bot, conn):
    result = bot.answer("is it available?", conn)

    assert result["intent"] == "availability"
    assert "include a catalog title" in result["message"]


# ---- the student's own loans ------------------------------------------------------
def test_my_loans_lists_loans_and_fines(bot, conn):
    student = _overdue_student(conn)
    result = bot.answer("show my loans", conn, student_id=student)

    assert result["intent"] == "my_loans" and result["data"]
    assert "Fines so far: KES" in result["message"]
    assert {"title", "due_on", "status", "fine"} <= set(result["data"][0])


def test_overdue_question_for_a_student_mentions_the_fine(bot, conn):
    student = _overdue_student(conn)
    result = bot.answer("do i have any overdue books", conn, student_id=student)

    assert result["intent"] == "overdue"
    assert "KES" in result["message"] and all(row["days_overdue"] > 0 for row in result["data"])


# ---- renewals: the chatbot acts, so these check the database -----------------------
def test_renewing_a_book_extends_the_due_date_once(bot, fresh_conn):
    student, loan_id, title = _clean_loan(fresh_conn)
    before = circulation.open_loans(fresh_conn, student).set_index("loan_id").loc[loan_id]

    first = bot.answer(f"Renew '{title}'", fresh_conn, student_id=student)
    after = circulation.open_loans(fresh_conn, student).set_index("loan_id").loc[loan_id]
    second = bot.answer(f"Renew '{title}'", fresh_conn, student_id=student)

    assert first["intent"] == "renew" and after["renewals"] == before["renewals"] + 1
    assert after["due_on"] > before["due_on"]
    assert first["message"].startswith("Renewed")
    assert "already been renewed" in second["message"]          # the rule allows one renewal only
    final = circulation.open_loans(fresh_conn, student).set_index("loan_id").loc[loan_id]
    assert final["renewals"] == after["renewals"] and final["due_on"] == after["due_on"]


def test_a_question_about_renewing_never_renews_anything(bot, fresh_conn):
    student, loan_id, _ = _clean_loan(fresh_conn)
    before = circulation.open_loans(fresh_conn, student).set_index("loan_id").loc[loan_id]

    result = bot.answer("can i renew a loan", fresh_conn, student_id=student)
    after = circulation.open_loans(fresh_conn, student).set_index("loan_id").loc[loan_id]

    assert result["intent"] == "policy" and "Renewals" in result["message"]
    assert after["renewals"] == before["renewals"] and after["due_on"] == before["due_on"]


def test_an_overdue_loan_cannot_be_renewed_through_the_chat(bot, fresh_conn):
    loans = circulation.open_loans(fresh_conn)
    late = loans[(loans["days_left"] < 0) & ~loans["title"].str.contains("['\",]")].iloc[0]
    student, title = int(late["student_id"]), late["title"]

    result = bot.answer(f"Renew '{title}'", fresh_conn, student_id=student)
    after = circulation.open_loans(fresh_conn, student).set_index("loan_id").loc[int(late["loan_id"])]

    assert after["renewals"] == late["renewals"] and after["due_on"] == late["due_on"]
    assert "overdue" in result["message"].lower()


def test_renew_without_a_selected_student_asks_to_select_one(bot, conn):
    result = bot.answer("please renew it", conn)

    assert "Select a student" in result["message"]


# ---- similar books -----------------------------------------------------------------
def test_similar_books_excludes_the_book_itself(bot, conn):
    result = bot.answer("books similar to Clean Code", conn)

    assert result["intent"] == "similar" and result["entities"]["book_title"] == "Clean Code"
    assert len(result["data"]) == 5
    assert "Clean Code" not in {row["title"] for row in result["data"]}


# ---- entity extraction: the two false matches we found and fixed -------------------
def test_a_topic_word_is_not_mistaken_for_a_book_title(bot):
    entities = bot.extract_entities("find books about statistics")

    assert entities["book_title"] is None


def test_a_topic_word_is_not_mistaken_for_an_author(bot):
    entities = bot.extract_entities("books about the history of kenya")

    assert entities["author"] is None


# ---- demand forecast with buying advice ----------------------------------------------
def test_forecast_question_about_copies_returns_buying_advice(bot, conn):
    result = bot.answer("which books need more copies", conn)

    assert result["intent"] == "forecast" and result["data"]
    assert {"copies", "forecast_loans", "copies_needed", "advice"} <= set(result["data"][0])


# ---- small talk, help and honesty about what it does not understand ------------------
@pytest.mark.parametrize("message, expected", [
    ("hello", "Hello"), ("thanks", "welcome"), ("goodbye", "Goodbye"),
])
def test_small_talk_gets_a_friendly_reply(bot, conn, message, expected):
    result = bot.answer(message, conn)

    assert result["intent"] == "smalltalk" and expected in result["message"]


def test_gibberish_gets_a_not_sure_reply_with_suggestions(bot, conn):
    result = bot.answer("asdf qwerty zxcv", conn)

    assert "not sure" in result["message"] and "Try:" in result["message"]
    assert result["data"] == []


def test_empty_message_is_rejected_politely(bot, conn):
    assert "enter a question" in bot.answer("   ", conn)["message"]


# ---- measured quality ----------------------------------------------------------------
def _normalised(sentence):
    return " ".join(sentence.lower().replace("'", " ").replace('"', " ").replace("?", " ").split())


def test_benchmark_sentences_were_never_used_for_training():
    trained = {_normalised(s) for examples in INTENT_EXAMPLES.values() for s in examples}
    benchmark = [s for s, _ in CHATBOT_DEV + CHATBOT_TEST]

    assert not [s for s in benchmark if _normalised(s) in trained]


def test_chatbot_meets_its_accuracy_targets_on_unseen_sentences(bot):
    dev = evaluate_chatbot(bot, CHATBOT_DEV)
    test = evaluate_chatbot(bot, CHATBOT_TEST)

    assert dev["strict_accuracy"] >= 0.80
    assert test["strict_accuracy"] >= 0.80
    assert test["accuracy_if_sure"] >= 0.85


# ---- answers that respect who is asking and what happens next -----------------------
def _plain_title(conn, available: bool):
    """A catalogue title that can be quoted in a message, with a free copy (or with all copies out)."""
    books = db.books_df(conn).merge(db.availability_df(conn)[["book_id", "available"]], on="book_id")
    wanted = books[(books["available"] > 0) == available]
    return wanted[~wanted["title"].str.contains("['\",:]")].iloc[0]


def test_yes_copies_free_is_not_misleading_for_a_student_who_cannot_borrow(bot, conn):
    student_id = _overdue_student(conn)
    title = _plain_title(conn, available=True)["title"]

    blocked = bot.answer(f"Is '{title}' available?", conn, student_id=student_id)
    anonymous = bot.answer(f"Is '{title}' available?", conn)

    assert blocked["message"].startswith("Yes.") and "can not borrow until you return" in blocked["message"]
    assert "can not borrow" not in anonymous["message"]


def test_a_book_that_is_fully_out_says_when_it_is_due_back(bot, conn):
    row = _plain_title(conn, available=False)
    outlook = circulation.return_outlook(conn, int(row["book_id"]))
    result = bot.answer(f"Is '{row['title']}' available?", conn)

    assert result["message"].startswith(("All", "The only copy"))
    assert result["data"][0]["available"] == 0
    if outlook["next_due"]:
        assert dates.short(outlook["next_due"]) in result["message"]
        assert result["data"][0]["next_due"] == outlook["next_due"].isoformat()


def test_late_copies_of_a_fully_out_book_are_mentioned(bot, conn):
    books = db.books_df(conn).set_index("title")
    for title in books.index:
        outlook = circulation.return_outlook(conn, int(books.loc[title, "book_id"]))
        stock = db.copies_available(conn, int(books.loc[title, "book_id"]))
        if stock == 0 and outlook["late_copies"] and outlook["next_due"] and "'" not in title and "," not in title:
            result = bot.answer(f"Is '{title}' available?", conn)
            assert "overdue" in result["message"]
            return
    pytest.fail("the simulated data should have a fully-out book with a late copy")


def test_my_loans_says_borrowing_is_paused_instead_of_offering_more(bot, conn):
    result = bot.answer("what do i owe", conn, student_id=_overdue_student(conn))

    assert result["intent"] == "my_loans"
    assert "Borrowing is paused until you return" in result["message"]
    assert "You can borrow" not in result["message"]


def test_the_chat_will_not_renew_while_another_book_is_overdue(bot, fresh_conn):
    student_id, _, loan_id, title = student_with_late_and_ok_loan(fresh_conn)
    before = db.get_loan(fresh_conn, loan_id)

    result = bot.answer(f"Renew '{title}'", fresh_conn, student_id=student_id)

    assert "can not be renewed while" in result["message"]
    assert db.get_loan(fresh_conn, loan_id)["due_on"] == before["due_on"]


# ---- recommendations by topic ----------------------------------------------------
def test_recommendations_can_be_limited_to_a_genre(bot, conn):
    # a heavy reader may already have borrowed every law book, so take the first student with something new
    results = (bot.answer("recommend law books", conn, student_id=student_id) for student_id in range(1, 20))
    result = next(r for r in results if r["data"])

    assert result["intent"] == "recommendations"
    assert {row["genre"] for row in result["data"]} == {"Law & Governance"}
    assert "Law & Governance" in result["message"]


@pytest.mark.parametrize("message, genre", [
    ("recommend computer science books", "Computer Science"),     # the longer phrase beats plain "science"
    ("any science books?", "Natural Sciences"),
    ("I want self-help books", "Self-Development"),
    ("books on Politics", "History & Politics"),
    ("recommend Law & Governance books", "Law & Governance"),
    ("recommend books about web development", None),               # "development" alone is not a genre
    ("recommend something", None),
])
def test_genre_words_are_recognised(bot, message, genre):
    assert bot.genre_in(message) == genre


# ---- real dates --------------------------------------------------------------------
def test_due_this_week_means_the_next_seven_days_even_on_a_saturday(bot, fresh_conn):
    saturday = date(2026, 11, 21)
    db.set_today(fresh_conn, saturday)

    result = bot.answer("what is due this week", fresh_conn)

    expected = db.due_loans_df(fresh_conn, saturday, saturday + timedelta(days=6))
    assert result["intent"] == "due_soon" and len(result["data"]) == len(expected) > 0
    assert "Sat 21 Nov" in result["message"] and "Fri 27 Nov" in result["message"]