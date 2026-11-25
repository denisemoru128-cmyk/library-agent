"""The docked assistant: how messages are drawn (time, ticks, the "understood as" line) and how the panel
behaves in the real app (questions, quick-reply chips, memory, clearing, one chat per student).
"""
import re

import pytest

from library_agent import circulation, db
from views import chat, signin


def _html(at) -> str:
    return " ".join(element.value for element in at.get("html"))


def _messages(at, student_id):
    return at.session_state["chat"][student_id]["messages"]


def _ask(at, text):
    at.chat_input(key="chat_input").set_value(text).run()
    assert not at.exception, [e.value for e in at.exception]


# ---- how a message is drawn -------------------------------------------------------------
def test_one_tick_means_sent_and_two_mean_understood():
    assert 'class="tick ok"' in chat._tick(True) and "Sent and understood" in chat._tick(True)
    assert 'class="tick "' in chat._tick(False) and "not understood" in chat._tick(False)


def test_the_bots_markdown_is_turned_into_html_and_everything_else_is_escaped():
    html = chat._rich("**Fines.** KES 20 a day\n- first <b>rule</b>\n- second")

    assert "<b>Fines.</b>" in html
    assert "<ul><li>first &lt;b&gt;rule&lt;/b&gt;</li><li>second</li></ul>" in html
    assert "<b>rule</b>" not in html                          # nothing a user typed can become markup


def test_a_book_in_an_answer_gets_one_short_status_line_whatever_the_record():
    late = {"title": "A", "due_on": "2026-11-10", "days_left": -4, "fine": 80}
    soon = {"title": "B", "due_on": "2026-11-20", "days_left": 6, "fine": 0}
    free = {"title": "C", "available": 2, "next_due": None}
    out = {"title": "D", "available": 0, "next_due": "2026-11-28"}
    forecast = {"title": "E", "advice": "Buy 1 more", "available": 0}

    assert chat._status(late) == ("Due Tue 10 Nov · 4 days late · KES 80", "late")
    assert chat._status(soon) == ("Due Fri 20 Nov · in 6 days", "")
    assert chat._status(free) == ("2 copies free", "ok")
    assert chat._status(out) == ("No free copy · next back Sat 28 Nov", "soon")
    assert chat._status(forecast) == ("Buy 1 more", "ok")


def test_a_long_list_is_cut_to_three_books_and_says_how_many_more():
    message = {"text": "Found 5 matching titles.", "time": "10:24", "intent": "search", "confidence": 0.99,
               "understood": True,
               "data": [{"title": f"Book {i}", "available": 1} for i in range(5)]}
    html = chat._bot(message, {})

    assert html.count('class="mini"') == 3 and "and 2 more" in html
    assert "understood as search · 99%" in html and "10:24" in html


def test_a_message_the_bot_was_unsure_about_says_so_instead_of_claiming_an_intent():
    message = {"text": "I am not sure what you mean.", "time": "10:24", "intent": "smalltalk", "confidence": 0.18,
               "understood": False, "data": []}
    html = chat._bot(message, {})

    assert "not sure what you meant · 18%" in html and "understood as" not in html


# ---- the panel in the real app ----------------------------------------------------------
def test_the_panel_is_on_every_student_screen_and_greets_by_first_name(app, conn):
    good, _ = signin.demo_students(conn)
    first_name = good["text"].split()[0]
    for section in ("Home", "Search", "My loans", "For you"):
        at = app(student_id=good["student_id"], section=section)
        assert at.chat_input(key="chat_input"), section
        assert f"Hello {first_name}" in _html(at), section


def test_the_librarian_has_no_chat_panel(app):
    at = app(role="Librarian")

    assert not at.chat_input


def test_a_question_is_answered_with_time_ticks_and_the_understood_as_line(app, conn):
    good, _ = signin.demo_students(conn)
    student_id = good["student_id"]
    at = app(student_id=student_id)
    _ask(at, "find books about statistics")
    page = _html(at)
    me, bot = _messages(at, student_id)[-2:]

    assert me["text"] == "find books about statistics" and bot["intent"] == "search"
    assert re.fullmatch(r"\d\d:\d\d", me["time"]) and bot["time"] == me["time"]
    assert 'class="tick ok"' in page                              # understood: two ticks
    assert re.search(r"understood as search · \d+%", page)
    assert "Found 5 matching titles." in page


def test_a_message_the_bot_does_not_understand_keeps_one_tick(app, conn):
    good, _ = signin.demo_students(conn)
    at = app(student_id=good["student_id"])
    _ask(at, "blorp zzz")
    me, bot = _messages(at, good["student_id"])[-2:]

    assert not me["understood"] and not bot["understood"]
    assert 'class="tick "' in _html(at) and "not sure what you meant" in _html(at)


def test_the_quick_replies_each_start_the_right_conversation(app, conn):
    good, _ = signin.demo_students(conn)
    student_id = good["student_id"]
    at = app(student_id=student_id)
    expected = {"What is due soon?": "due_soon", "My loans and fines": "my_loans",
                "Recommend a book": "recommendations", "Library rules": "policy"}
    assert set(expected) == set(chat.START_CHIPS)
    for text, intent in expected.items():
        at.button(key="chat_clear").click().run()            # a fresh chat always offers the starting chips
        at.button(key=f"chip_{text}").click().run()
        answer = _messages(at, student_id)[-1]
        assert answer["intent"] == intent and answer["understood"], text


def test_after_a_book_answer_the_chips_follow_the_conversation_and_use_its_memory(app, conn):
    good, _ = signin.demo_students(conn)
    student_id = good["student_id"]
    at = app(student_id=student_id)
    _ask(at, "find books about statistics")
    assert [b.label for b in at.button if b.key.startswith("chip_")] == chat.BOOK_CHIPS

    at.button(key="chip_Is it available?").click().run()
    available = _messages(at, student_id)[-1]
    assert available["intent"] == "availability" and available["text"].startswith("Assuming you mean")

    at.button(key="chip_Books similar to it").click().run()
    similar = _messages(at, student_id)[-1]
    assert similar["intent"] == "similar" and similar["understood"]


def test_renewing_in_the_chat_changes_the_screen_too(app, conn):
    good, _ = signin.demo_students(conn)
    student_id = good["student_id"]
    loans = circulation.open_loans(conn, student_id).sort_values("days_left").head(3)
    loan = loans[loans["renewals"] == 0].iloc[0]
    at = app(student_id=student_id)
    assert not at.button(key=f"home_renew_{int(loan['loan_id'])}").disabled

    _ask(at, f"renew {loan['title']}")

    assert _messages(at, student_id)[-1]["intent"] == "renew"
    private = db.connect(at.db)
    try:
        assert db.get_loan(private, int(loan["loan_id"]))["renewals"] == 1
    finally:
        private.close()
    assert at.button(key=f"home_renew_{int(loan['loan_id'])}").disabled     # the card behind the chat caught up


def test_clear_starts_the_chat_again_with_just_the_greeting(app, conn):
    good, _ = signin.demo_students(conn)
    student_id = good["student_id"]
    at = app(student_id=student_id)
    _ask(at, "find books about statistics")
    assert len(_messages(at, student_id)) == 3                       # greeting, question, answer

    at.button(key="chat_clear").click().run()

    assert len(_messages(at, student_id)) == 1 and "Hello" in _messages(at, student_id)[0]["text"]


def test_each_student_has_their_own_conversation(app, conn):
    good, late = signin.demo_students(conn)
    at = app(student_id=good["student_id"])
    _ask(at, "find books about statistics")

    at.session_state["user"] = {"role": "student", "student_id": late["student_id"]}
    at.run()

    assert len(_messages(at, late["student_id"])) == 1                # only the greeting
    assert len(_messages(at, good["student_id"])) == 3                # the first student's chat is untouched
    assert "statistics" not in _html(at)