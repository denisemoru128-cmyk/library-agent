"""The assistant panel, docked on the right of every student screen.

It is a chat with the library chatbot (library_agent/chatbot.py). Each message shows its time and ticks:
one tick means "sent", two ticks mean "sent and understood", so a message the bot was not confident about
(below CHATBOT_MIN_CONFIDENCE) keeps a single grey tick. Under each answer a small line says how the
bot understood the question and how sure it was, which is the intent classifier showing its work.

The conversation (and the book the bot remembers) is kept per student in the session, so switching user
starts a fresh chat.
"""
import html
import re
from datetime import date, datetime

import streamlit as st

from library_agent import dates

from . import components as ui
from . import theme

START_CHIPS = ["What is due soon?", "My loans and fines", "Recommend a book", "Library rules"]
BOOK_CHIPS = ["Is it available?", "Books similar to it", "Recommend a book"]
BOOK_INTENTS = {"search", "availability", "similar", "recommendations", "author_search"}
MAX_BOOKS_SHOWN = 3


# ------------------------------------------------------------------ small pieces of HTML
def _now() -> str:
    return datetime.now().strftime("%H:%M")


def _tick(understood: bool) -> str:
    """One tick: sent. Two ticks: sent and understood. (Drawn with CSS, because st.html removes inline SVG.)"""
    label = "Sent and understood" if understood else "Sent, but not understood"
    return f'<span class="tick {"ok" if understood else ""}" role="img" aria-label="{label}" title="{label}"></span>'


def _rich(text: str) -> str:
    """The bot writes **bold** and '- ' bullet lines; turn those into HTML (everything else is escaped)."""
    blocks, items = [], []

    def flush():
        if items:
            blocks.append("<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>")
            items.clear()

    for line in text.split("\n"):
        safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", ui.esc(line))
        if safe.startswith("- "):
            items.append(safe[2:])
        else:
            flush()
            blocks.append(f"<p>{safe}</p>")
    flush()
    return "".join(blocks)


def _stock_line(available: int, next_due: str | None) -> tuple[str, str]:
    if available > 0:
        return ui.availability_line(available)
    return "No free copy" + (f" · next back {dates.short(date.fromisoformat(next_due[:10]))}" if next_due else ""), "soon"


def _status(record: dict) -> tuple[str, str]:
    """The one short line under a book in an answer, whatever kind of record the bot returned."""
    if record.get("due_on") and "days_left" in record:                       # one of my loans
        due = f"Due {dates.short(date.fromisoformat(str(record['due_on'])[:10]))}"
        if record["days_left"] < 0:
            return f"{due} · {ui.plural(-int(record['days_left']), 'day')} late · KES {record.get('fine', 0)}", "late"
        return f"{due} · {ui.days_left_text(int(record['days_left']))}", ""
    if record.get("due_on"):                                                  # due this week (library-wide view)
        return f"Due {dates.short(date.fromisoformat(str(record['due_on'])[:10]))}", ""
    if "advice" in record:                                                    # demand forecast
        return str(record["advice"]), "ok"
    if record.get("available") is not None:
        return _stock_line(int(record["available"]), record.get("next_due"))
    return "", ""


def _mini_book(record: dict, genres: dict) -> str:
    title = record["title"]
    text, tone = _status(record)
    return (f'<div class="mini">{theme.cover(title, genres.get(title, ""), "", 34)}'
            f'<div><b>{ui.esc(title)}</b><span class="{tone}">{ui.esc(text)}</span></div></div>')


def _me(message: dict) -> str:
    return (f'<div class="msg me"><div class="bub">{ui.esc(message["text"])}</div>'
            f'<div class="meta">{message["time"]} {_tick(message["understood"])}</div></div>')


def _bot(message: dict, genres: dict) -> str:
    parts = [f'<div class="msg bot"><div class="bub">{_rich(message["text"])}</div>']
    books = [r for r in message.get("data", []) if r.get("title")]
    if books:
        parts.append('<div class="minis">' + "".join(_mini_book(r, genres) for r in books[:MAX_BOOKS_SHOWN]) + "</div>")
        if len(books) > MAX_BOOKS_SHOWN:
            parts.append(f'<div class="more">and {len(books) - MAX_BOOKS_SHOWN} more</div>')
    if "intent" in message:
        how = (f"understood as {message['intent'].replace('_', ' ')}" if message["understood"]
               else "not sure what you meant")
        parts.append(f'<div class="und">{how} · {message["confidence"]:.0%}</div>')
    parts.append(f'<div class="meta">{message["time"]}</div></div>')
    return "".join(parts)


# ------------------------------------------------------------------ conversation state
def _chat_for(student_id: int, first_name: str) -> dict:
    chat = st.session_state.setdefault("chat", {}).setdefault(student_id, {"messages": [], "context": {}})
    if not chat["messages"]:
        chat["messages"].append({
            "who": "bot", "time": _now(),
            "text": f"Hello {first_name}. Ask about a book, your loans or the library rules."})
    return chat


def _suggestions(chat: dict) -> list[str]:
    last = next((m for m in reversed(chat["messages"]) if m["who"] == "bot" and "intent" in m), None)
    if last and last["understood"] and last["intent"] in BOOK_INTENTS and chat["context"].get("last_book"):
        return BOOK_CHIPS
    return START_CHIPS


def _ask_later(text: str) -> None:
    st.session_state["chat_pending"] = text


def _clear(student_id: int) -> None:
    st.session_state.get("chat", {}).pop(student_id, None)


def ask(conn, student_id: int, models, chat: dict, question: str) -> None:
    """Send one question to the chatbot and keep both lines of the conversation."""
    answer = models.chatbot.answer(question, conn, student_id=student_id, context=chat["context"])
    now = _now()
    chat["messages"].append({"who": "me", "text": question, "time": now, "understood": answer["understood"]})
    chat["messages"].append({"who": "bot", "text": answer["message"], "data": answer["data"], "time": now,
                             "intent": answer["intent"], "confidence": answer["confidence"],
                             "understood": answer["understood"]})


# ------------------------------------------------------------------ the panel
def render(conn, student_id: int, models) -> None:
    name = models.students.loc[models.students["student_id"] == student_id, "name"].iloc[0]
    chat = _chat_for(student_id, name.split()[0])
    genres = dict(zip(models.books["title"], models.books["genre"]))

    with st.container(key="chat_panel"):
        head = st.container(key="chat_head")
        log = st.container(key="chat_log")
        chips = st.container(key="chat_chips")
        entry = st.container(key="chat_entry")

    with entry:
        typed = st.chat_input("Ask the library assistant", key="chat_input")
    question = st.session_state.pop("chat_pending", None) or typed
    if question:
        ask(conn, student_id, models, chat, question)
        st.rerun()           # also refreshes the screen, because a renewal in the chat changes the numbers

    with head:
        title, clear = st.columns([3, 1], vertical_alignment="center")
        title.html('<div class="chat-h"><b>Assistant</b><small>Books, your loans and the rules</small></div>')
        clear.button("Clear", key="chat_clear", type="tertiary", on_click=_clear, args=(student_id,))
    with log:
        with st.container(key="chat_msgs"):        # one child inside the log, so it can stay scrolled to the bottom
            st.html("".join(_me(m) if m["who"] == "me" else _bot(m, genres) for m in chat["messages"]))
    with chips:
        for column, text in zip(st.columns(2), _suggestions(chat)[:2]):
            column.button(text, key=f"chip_{text}", on_click=_ask_later, args=(text,))
        for column, text in zip(st.columns(2), _suggestions(chat)[2:4]):
            column.button(text, key=f"chip_{text}", on_click=_ask_later, args=(text,))