"""The library chatbot.

Each message goes through four steps:

1. UNDERSTAND the intent. A scikit-learn classifier (TF-IDF + logistic
   regression) trained on the labelled sentences in chat_data.py predicts what
   the student wants (search, availability, renew, policy ...) and how sure it is.
   Because it learns from examples, it copes with wordings it has never seen.
2. EXTRACT entities: a book title, an author, a student number.
3. REMEMBER the conversation: the last book mentioned is kept in a ``context``
   dictionary, so "Is it available?" after a search means the book just found.
4. ANSWER by calling the tested modules (search, recommender, circulation,
   forecaster). The chatbot itself contains no library rules; it only routes.

If the classifier is not confident enough (config.CHATBOT_MIN_CONFIDENCE) the
bot says so and suggests questions it can answer, instead of guessing.
"""
import re
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline

from . import circulation, db, dates, nlp
from .chat_data import FAQ, GENRE_WORDS, INTENT_EXAMPLES
from .config import CHATBOT_MIN_CONFIDENCE, FORECAST_HORIZON_WEEKS, MAX_ACTIVE_LOANS

FAQ_MIN_SIMILARITY = 0.25   # below this the question does not match any rule entry

BOOK_TITLE_PATTERN = re.compile(r"""["']([^"']{2,})["']""")
STUDENT_ID_PATTERN = re.compile(r"\bstudent(?:\s*(?:id|number))?\s*#?\s*(\d+)\b", re.I)
NAME_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)*")
AUTHOR_QUERY_CUES = frozenset({
    "book", "books", "title", "titles", "available", "availability", "borrow",
    "borrowable", "stock", "written", "wrote", "works",
})
BUY_CUES = frozenset({"copies", "buy", "purchase", "acquire", "order"})
SIMILAR_CUES = frozenset({"similar", "like", "resemble", "resembling"})
QUESTION_STARTS = frozenset({"can", "could", "may", "how", "is", "are", "do", "does", "will"})
POINTING_WORDS = frozenset({"it", "this", "that", "one"})
SUGGESTIONS = ("Try: “Is Clean Code available?”, “Find books about statistics”, "
               "“What is due this week?”, “Recommend a book”, "
               "“What is the fine for late books?” or “Renew Clean Code”.")


@dataclass
class Turn:
    """Everything a handler needs to know about the message being answered."""
    query: str
    conn: object
    student_id: int | None
    context: dict
    intent: str
    confidence: float
    entities: dict


class LibraryChatbot:
    """Classify a library question and dispatch it to tested service modules."""

    def __init__(self, books: pd.DataFrame, search_engine, recommender, forecaster):
        self.books = books.reset_index(drop=True)
        self.search_engine = search_engine
        self.recommender = recommender
        self.forecaster = forecaster

        texts, labels = [], []
        for intent, examples in INTENT_EXAMPLES.items():
            texts.extend(examples)
            labels.extend([intent] * len(examples))
        self.classifier = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0, C=100),
        )
        self.classifier.fit(texts, labels)

        # Words used in titles, genres and descriptions are TOPICS, not author names:
        # otherwise "books about kenya" would match the author "Republic of Kenya".
        self._topic_words = {
            word for text in (self.books["title"] + " " + self.books["genre"] + " " + self.books["description"])
            for word in nlp.raw_words(text)
        }

        # FAQ retrieval: every phrasing of every rule question, remembered with its entry
        self._faq_owner, faq_texts = [], []
        for index, entry in enumerate(FAQ):
            for question in entry["questions"]:
                faq_texts.append(nlp.normalize(question))
                self._faq_owner.append(index)
        self._faq_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._faq_matrix = self._faq_vectorizer.fit_transform(faq_texts)

    # ------------------------------------------------------------ understanding
    def classify(self, query: str) -> tuple[str, float]:
        """Return the most likely intent and its model probability."""
        entities = self.extract_entities(query)
        if entities["author"] and AUTHOR_QUERY_CUES.intersection(nlp.raw_words(query)):
            return "author_search", 1.0
        title = entities["book_title"]
        if title:
            query = re.sub(re.escape(title), "book", query, flags=re.IGNORECASE)
        probabilities = self.classifier.predict_proba([query])[0]
        position = int(probabilities.argmax())
        intent, confidence = str(self.classifier.classes_[position]), float(probabilities[position])
        # Rule on top of the model: a named book + "similar"/"like" means "find similar books".
        if title and intent in {"recommendations", "search"} and SIMILAR_CUES.intersection(nlp.raw_words(query)):
            return "similar", max(confidence, CHATBOT_MIN_CONFIDENCE)
        return intent, confidence

    @staticmethod
    def _is_topic_mention(title: str, cleaned_query: str) -> bool:
        """'books about statistics' mentions a TOPIC that happens to be a one-word title.
        Without this check the book *Statistics* would hijack every statistics search."""
        if len(title.split()) > 1:
            return False
        word = re.escape(nlp.clean(title))
        return bool(re.search(rf"\b(?:about|on|in|of|for|regarding|around|into)\s+{word}\b", cleaned_query))

    def extract_entities(self, query: str) -> dict:
        """Extract a catalog title, author, and optional student identifier."""
        titles = self.books["title"].tolist()
        normalized_query = nlp.normalize(query)
        cleaned_query = nlp.clean(query)
        quoted = BOOK_TITLE_PATTERN.search(query)
        matched_title = None

        if quoted:
            quoted_normalized = nlp.normalize(quoted.group(1))
            for title in titles:
                if nlp.normalize(title) == quoted_normalized:
                    matched_title = title
                    break

        if matched_title is None:
            matches = [
                title for title in titles
                if nlp.normalize(title) and nlp.normalize(title) in normalized_query
                and not self._is_topic_mention(title, cleaned_query)
            ]
            if matches:
                matched_title = max(matches, key=lambda title: len(nlp.normalize(title)))

        query_name_words = {
            word.replace("'", "").replace("’", "")
            for word in NAME_WORD_PATTERN.findall(cleaned_query)
            if len(word.replace("'", "").replace("’", "")) >= 4
        }
        author_matches = []
        for author in self.books["author"].dropna().unique():
            author_words = {
                word.replace("'", "").replace("’", "")
                for word in NAME_WORD_PATTERN.findall(nlp.clean(author))
                if len(word.replace("'", "").replace("’", "")) >= 4 and word not in self._topic_words
            }
            matched_words = query_name_words & author_words
            if matched_words:
                author_matches.append((len(matched_words), sum(map(len, matched_words)), author))
        matched_author = max(author_matches, default=(0, 0, None))[2]

        entities = {"book_title": matched_title, "author": matched_author}
        student = STUDENT_ID_PATTERN.search(query)
        if student:
            entities["student_id"] = int(student.group(1))
        return entities

    def genre_in(self, query: str) -> str | None:
        """The catalogue genre a message names ("recommend law books" -> Law & Governance), or None.
        The longest matching phrase wins, so "computer science" is not read as plain "science"."""
        text = " " + re.sub(r"[^a-z0-9\- ]+", " ", nlp.clean(query).replace("&", " and ")) + " "
        known = set(self.books["genre"])
        best, best_length = None, 0
        for genre in known:
            for phrase in {genre.lower().replace("&", "and"), *GENRE_WORDS.get(genre, [])}:
                if f" {phrase} " in text and len(phrase) > best_length:
                    best, best_length = genre, len(phrase)
        return best

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _records(frame: pd.DataFrame) -> list[dict]:
        if frame.empty:
            return []
        return frame.where(pd.notna(frame), None).to_dict(orient="records")

    @staticmethod
    def _response(intent: str, confidence: float, message: str, data: list, entities: dict) -> dict:
        return {
            "intent": intent,
            "confidence": confidence,
            "message": message,
            "data": data,
            "entities": entities,
        }

    def _reply(self, turn: Turn, message: str, data: list | None = None) -> dict:
        return self._response(turn.intent, turn.confidence, message, data or [], turn.entities)

    def _book_in_focus(self, turn: Turn) -> tuple[str | None, str]:
        """The book the student means: named in this message, or remembered from earlier.
        Returns (title, prefix) where prefix tells the student when we assumed it."""
        if turn.entities["book_title"]:
            return turn.entities["book_title"], ""
        remembered = turn.context.get("last_book")
        if remembered:
            return remembered, f"Assuming you mean “{remembered}”. "
        return None, ""

    def _book_row(self, title: str) -> pd.Series:
        return self.books[self.books["title"] == title].iloc[0]

    @staticmethod
    def _remember(context: dict, entities: dict, response: dict) -> None:
        """Keep the most recent book so later questions can say 'it'."""
        if entities.get("book_title"):
            context["last_book"] = entities["book_title"]
        elif response["intent"] in {"search", "recommendations", "author_search"} and response["data"]:
            first = response["data"][0]
            if first.get("title"):
                context["last_book"] = first["title"]

    # ------------------------------------------------------------------ answer
    def answer(self, query: str, conn, student_id: int | None = None, context: dict | None = None) -> dict:
        """Answer one message.

        ``context`` is a dictionary the caller keeps between messages (for example in
        Streamlit's session state). The bot reads and updates it to remember the last
        book discussed. Without it the bot simply answers each message on its own.
        """
        query = query.strip()
        if not query:
            return self._response("unknown", 0.0, "Please enter a question.", [], {})
        context = {} if context is None else context

        intent, confidence = self.classify(query)
        entities = self.extract_entities(query)
        if confidence < CHATBOT_MIN_CONFIDENCE:
            return self._response(
                intent, confidence, f"I am not sure what you mean. {SUGGESTIONS}", [], entities)

        turn = Turn(query, conn, student_id, context, intent, confidence, entities)
        handler = getattr(self, f"_handle_{intent}", self._handle_help)
        response = handler(turn)
        self._remember(context, entities, response)
        return response

    # ---------------------------------------------------------------- handlers
    def _handle_availability(self, turn: Turn) -> dict:
        title, prefix = self._book_in_focus(turn)
        if title is None:
            return self._reply(turn, "Please include a catalog title, for example: Is 'Clean Code' available?")
        book = self._book_row(title)
        book_id = int(book["book_id"])
        count = int(db.availability_df(turn.conn).set_index("book_id").loc[book_id, "available"])
        record = {"book_id": book_id, "title": title, "available": count, "next_due": None}
        if count > 0:
            message = f"Yes. {title} has {count} available cop{'y' if count == 1 else 'ies'}."
            message += self._eligibility_note(turn)
        else:
            outlook = circulation.return_outlook(turn.conn, book_id)
            message = self._all_out_message(title, int(book["copies"]), outlook)
            record["next_due"] = outlook["next_due"].isoformat() if outlook["next_due"] else None
        return self._reply(turn, prefix + message, [record])

    @staticmethod
    def _all_out_message(title: str, copies: int, outlook: dict) -> str:
        """Say when a fully borrowed book is expected back, instead of a bare 'not available'."""
        if outlook["copies_out"] == 0:
            return f"No copies of {title} are available: the library has none on the shelf."
        owned = "The only copy is out" if copies == 1 else f"All {copies} copies are out"
        late = outlook["late_copies"]
        late_note = (f" {late} {'copy is' if late == 1 else 'copies are'} already overdue." if late else "")
        if outlook["next_due"] is None:
            return f"{owned}, and {'it is' if copies == 1 else 'every copy is'} overdue, so there is no return date yet."
        return f"{owned}. The next is due back on {dates.short(outlook['next_due'])}.{late_note}"

    @staticmethod
    def _eligibility_note(turn: Turn) -> str:
        """Add what stops THIS student from borrowing, so "Yes, 3 copies" is never misleading."""
        if turn.student_id is None:
            return ""
        summary = circulation.student_summary(turn.conn, turn.student_id)
        if summary["borrowing_paused"]:
            return f" But you can not borrow until you return “{summary['blocking_title']}”."
        if summary["slots_free"] == 0:
            return f" But you already have {MAX_ACTIVE_LOANS} books out. Return one first."
        return ""

    def _handle_due_soon(self, turn: Turn) -> dict:
        # A rolling window: "this week" is the next 7 days, "next week" the 7 after that. (A calendar
        # week would leave a student asking on a Saturday with just two days.)
        as_of = db.today(turn.conn)
        start = as_of + timedelta(days=7 if "next week" in turn.query.lower() else 0)
        end = start + timedelta(days=6)
        result = db.due_loans_df(turn.conn, start=start, end=end, student_id=turn.student_id)
        return self._reply(turn, f"Found {len(result)} loan(s) due between {dates.short(start)} and {dates.short(end)}.",
                           self._records(result))

    def _handle_overdue(self, turn: Turn) -> dict:
        overdue = circulation.overdue_loans(turn.conn)
        if turn.student_id is not None:
            overdue = overdue[overdue["student_id"] == turn.student_id]
        result = overdue[["loan_id", "student_name", "reg_no", "title", "due_on", "days_overdue", "fine"]].copy()
        result["due_on"] = result["due_on"].dt.strftime("%Y-%m-%d")
        if turn.student_id is not None:
            owed = int(result["fine"].sum())
            message = (f"You have {len(result)} overdue book(s). Fines so far: KES {owed}."
                       if len(result) else "You have no overdue books.")
        else:
            message = f"Found {len(result)} overdue loan(s)."
        return self._reply(turn, message, self._records(result.reset_index(drop=True)))

    def _handle_recommendations(self, turn: Turn) -> dict:
        if turn.student_id is None:
            return self._reply(turn, "Select a student first so I can personalize recommendations.")
        genre = self.genre_in(turn.query)
        result = self.recommender.recommend(turn.student_id, top_n=5, availability=db.availability_df(turn.conn),
                                            genre=genre)
        if genre and result.empty:
            return self._reply(turn, f"I have no new {genre} books to suggest: you have borrowed them all.")
        what = f"{genre} picks" if genre else "recommendations"
        return self._reply(turn, f"Here are {len(result)} {what} for you.", self._records(result))

    def _handle_search(self, turn: Turn) -> dict:
        result = self.search_engine.search(turn.query, top_n=5, availability=db.availability_df(turn.conn))
        return self._reply(turn, f"Found {len(result)} matching title(s).", self._records(result))

    def _handle_author_search(self, turn: Turn) -> dict:
        author = turn.entities["author"]
        if author is None:
            return self._reply(turn, "Please include an author's name so I can find their books.")
        result = self.books.loc[
            self.books["author"] == author, ["book_id", "title", "author"]
        ].merge(db.availability_df(turn.conn)[["book_id", "available"]], on="book_id")
        available_only = bool(
            {"available", "availability", "borrow", "borrowable", "stock"}.intersection(nlp.raw_words(turn.query))
        )
        if available_only:
            result = result[result["available"] > 0]
            message = f"Found {len(result)} available book(s) by {author}."
        else:
            message = f"Found {len(result)} book(s) by {author}; availability is shown."
        return self._reply(turn, message, self._records(result))

    def _handle_forecast(self, turn: Turn) -> dict:
        if BUY_CUES.intersection(nlp.raw_words(turn.query)):
            top = self.forecaster.high_demand_books(top_n=5, availability=db.availability_df(turn.conn))
            columns = ["title", "copies", "available", "forecast_loans", "copies_needed", "advice"]
            return self._reply(
                turn, "These books are under the most pressure over the next 4 weeks, with my buying advice.",
                self._records(top[columns]))
        result = self.forecaster.forecast(by="genre", horizon_weeks=FORECAST_HORIZON_WEEKS)
        ranked = (
            result.groupby("genre", as_index=False)["predicted_loans"].sum()
            .sort_values("predicted_loans", ascending=False).head(5)
        )
        return self._reply(turn, "These genres have the highest predicted borrowing demand.", self._records(ranked))

    def _handle_policy(self, turn: Turn) -> dict:
        scores = cosine_similarity(self._faq_vectorizer.transform([nlp.normalize(turn.query)]), self._faq_matrix)[0]
        best = int(scores.argmax())
        if scores[best] >= FAQ_MIN_SIMILARITY:
            entry = FAQ[self._faq_owner[best]]
            return self._reply(turn, f"**{entry['topic']}.** {entry['answer']}")
        summary = "\n".join(f"- **{entry['topic']}.** {entry['answer']}" for entry in FAQ)
        return self._reply(turn, "Here are the library rules:\n" + summary)

    def _handle_my_loans(self, turn: Turn) -> dict:
        if turn.student_id is None:
            return self._reply(turn, "Select a student first so I can show your loans.")
        summary = circulation.student_summary(turn.conn, turn.student_id)
        loans = circulation.open_loans(turn.conn, turn.student_id)
        if loans.empty:
            return self._reply(turn, f"You have no books on loan. You can borrow up to {MAX_ACTIVE_LOANS}.")
        table = loans[["loan_id", "title", "due_on", "days_left", "status", "fine"]].copy()
        table["due_on"] = table["due_on"].dt.strftime("%Y-%m-%d")
        message = (f"You have {summary['active_loans']} book(s) on loan"
                   + (f", {summary['overdue']} overdue" if summary["overdue"] else "")
                   + f". Fines so far: KES {summary['fines_owed']}. "
                   + (f"Borrowing is paused until you return “{summary['blocking_title']}”."
                      if summary["borrowing_paused"] else f"You can borrow {summary['can_borrow_more']} more."))
        return self._reply(turn, message, self._records(table))

    def _handle_renew(self, turn: Turn) -> dict:
        words = nlp.raw_words(turn.query)
        asks_about_rules = (words and words[0] in QUESTION_STARTS and not turn.entities["book_title"]
                            and not POINTING_WORDS.intersection(words))
        if asks_about_rules:  # a question like "can i renew a loan?" must never renew anything by itself
            turn.intent = "policy"  # report what we really did: answer from the rules
            return self._handle_policy(turn)
        if turn.student_id is None:
            return self._reply(turn, "Select a student first so I can renew your loan.")
        loans = circulation.open_loans(turn.conn, turn.student_id)
        title, prefix = self._book_in_focus(turn)
        if title is None:
            if loans.empty:
                return self._reply(turn, "You have no books on loan, so there is nothing to renew.")
            if len(loans) > 1:
                names = ", ".join(f"“{t}”" for t in loans["title"])
                return self._reply(turn, f"Which book would you like to renew? You have: {names}.")
            title = loans.iloc[0]["title"]
        match = loans[loans["title"] == title]
        if match.empty:
            return self._reply(turn, f"You do not have “{title}” on loan, so there is nothing to renew.")
        outcome = circulation.renew_loan(turn.conn, int(match.iloc[0]["loan_id"]))
        turn.context["last_book"] = title
        return self._reply(turn, prefix + outcome.message)

    def _handle_similar(self, turn: Turn) -> dict:
        title, prefix = self._book_in_focus(turn)
        if title is None:
            return self._reply(turn, "Which book? For example: books similar to 'Clean Code'.")
        result = self.recommender.similar_books(int(self._book_row(title)["book_id"]), top_n=5)
        result = result.merge(db.availability_df(turn.conn)[["book_id", "available"]], on="book_id")
        return self._reply(turn, f"{prefix}Books similar to “{title}”:", self._records(result))

    def _handle_smalltalk(self, turn: Turn) -> dict:
        words = set(nlp.raw_words(turn.query))
        if words & {"thanks", "thank", "cheers"}:
            return self._reply(turn, "You are welcome! Ask me anything else about the library.")
        if words & {"bye", "goodbye", "later"}:
            return self._reply(turn, "Goodbye! Happy reading.")
        return self._reply(turn, "Hello! I am the library assistant. I can find books, check availability, "
                                 "recommend what to read and manage your loans.")

    def _handle_unknown(self, turn: Turn) -> dict:
        return self._reply(
            turn, "I can help with library search, recommendations, availability, your loans, renewals, "
                  "library rules and demand forecasts.")

    def _handle_help(self, turn: Turn) -> dict:
        return self._reply(
            turn, "I can search the catalog, recommend books, check availability, show your loans and fines, "
                  "renew a book, explain the library rules, find similar books and forecast demand. "
                  + SUGGESTIONS)