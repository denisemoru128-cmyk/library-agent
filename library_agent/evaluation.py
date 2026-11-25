"""Measuring how well the AI parts work.

Each ``evaluate_*`` function returns plain numbers, so the same code feeds the
automated tests, the ``evaluate.py`` command and the report. A model with no
measured results is only a claim; these functions turn it into evidence.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

from .forecast import DemandForecaster
from .recommender import METHODS, Recommender
from .search import BookSearch

# (what a student might type, titles that would be a correct answer)
SEARCH_BENCHMARK = [
    ("machine learning with python", ["Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow"]),
    ("intro to algorithms", ["Introduction to Algorithms"]),
    ("learn sql and databases", ["Database System Concepts"]),
    ("how computers communicate over a network", ["Computer Networking: A Top-Down Approach"]),
    ("neural networks", ["Deep Learning", "Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow"]),
    ("probability theory", ["A First Course in Probability"]),
    ("calculus", ["Calculus: Early Transcendentals"]),
    ("circuits", ["Fundamentals of Electric Circuits", "Microelectronic Circuits"]),
    ("economics", ["Principles of Economics", "Microeconomics"]),
    ("marketing strategy", ["Principles of Marketing"]),
    ("human anatomy", ["Gray's Anatomy for Students"]),
    ("infectious disease bacteria viruses", ["Medical Microbiology"]),
    ("kenya constitution", ["The Constitution of Kenya, 2010"]),
    ("criminal law", ["Textbook on Criminal Law"]),
    ("land ownership law", ["Elements of Land Law"]),
    ("chinua achebe", ["Things Fall Apart"]),
    ("ngugi", ["A Grain of Wheat", "Weep Not, Child", "Petals of Blood", "Decolonising the Mind",
               "The River Between"]),
    ("orwell dystopia", ["Nineteen Eighty-Four", "Animal Farm"]),
    ("pride and prejudice", ["Pride and Prejudice"]),
    ("mau mau rebellion", ["Imperial Reckoning", "A Grain of Wheat"]),
    ("nelson mandela autobiography", ["Long Walk to Freedom"]),
    ("black holes and the universe", ["A Brief History of Time", "Cosmos"]),
    ("evolution genes", ["The Selfish Gene"]),
    ("pesticides pollution", ["Silent Spring"]),
    ("soil agriculture", ["The Nature and Properties of Soils"]),
    ("building good habits", ["Atomic Habits", "The 7 Habits of Highly Effective People"]),
    ("how to focus", ["Deep Work"]),
    ("effective study techniques memory", ["Make It Stick", "A Mind for Numbers"]),
    ("investing money", ["The Intelligent Investor", "Rich Dad Poor Dad"]),
    ("startup business experiments", ["The Lean Startup"]),
    # queries with typing mistakes
    ("machin lerning", ["Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow",
                        "An Introduction to Statistical Learning"]),
    ("pythn programing", ["Python Crash Course"]),
    ("thinkin fast and slow", ["Thinking, Fast and Slow"]),
    ("chemestry", ["Chemistry: The Central Science", "Organic Chemistry"]),
    ("habbits", ["Atomic Habits", "The 7 Habits of Highly Effective People"]),
]


def evaluate_search(books, k: int = 3) -> dict:
    """Hit-rate@k: how often a correct book appears in the top k results."""
    engine = BookSearch(books)
    hits, misses = 0, []
    for query, correct_titles in SEARCH_BENCHMARK:
        top = engine.search(query, top_n=k)["title"].tolist()
        if any(t in correct_titles for t in top):
            hits += 1
        else:
            misses.append((query, top))
    return {"queries": len(SEARCH_BENCHMARK), "hits": hits, "hit_rate": hits / len(SEARCH_BENCHMARK),
            "k": k, "misses": misses}


def evaluate_recommender(books, students, loans, as_of, test_days: int = 60, k: int = 10,
                         min_train: int = 5, max_history: int | None = None, seed: int = 0) -> dict:
    """Predict the future from the past.

    The recommender only sees loans up to a cut-off date. We then check whether
    the books each student *really borrowed afterwards* appear in their top-k.
      hit rate  = share of students with at least one correct book in the top k
      recall    = share of each student's future books found in the top k (averaged)
    ``max_history`` limits how many early loans per student the model may see,
    which simulates new students with little history (cold start).
    """
    as_of = pd.Timestamp(as_of)
    cutoff = as_of - timedelta(days=test_days)
    train = loans[loans["borrowed_on"] <= cutoff]
    test = loans[loans["borrowed_on"] > cutoff]
    if max_history:
        train = train.sort_values("borrowed_on").groupby("student_id").head(max_history)
        min_train = min(min_train, max_history)

    model = Recommender(books, students, train, cutoff)
    rng = np.random.default_rng(seed)
    all_ids = books["book_id"].to_numpy()
    scores = {m: {"hits": 0, "recall": 0.0} for m in (*METHODS, "random")}
    n_students = 0

    for sid, group in test.groupby("student_id"):
        seen = set(train.loc[train["student_id"] == sid, "book_id"])
        wanted = set(group["book_id"]) - seen
        if len(seen) < min_train or not wanted:
            continue
        n_students += 1
        for method in METHODS:
            top = set(model.recommend(sid, top_n=k, method=method)["book_id"])
            scores[method]["hits"] += bool(top & wanted)
            scores[method]["recall"] += len(top & wanted) / len(wanted)
        pool = [b for b in all_ids if b not in seen]
        top = set(rng.choice(pool, size=k, replace=False))
        scores["random"]["hits"] += bool(top & wanted)
        scores["random"]["recall"] += len(top & wanted) / len(wanted)

    return {
        "students_tested": n_students, "k": k, "test_days": test_days,
        "methods": {m: {"hit_rate": s["hits"] / n_students, "recall": s["recall"] / n_students}
                    for m, s in scores.items()},
    }

def evaluate_forecast(books, loans, as_of, origins: int = 6, horizon: int = 4) -> dict:
    """Backtest the demand forecaster: pretend it is some weeks ago, forecast, compare with reality.

    ``origins`` different past Mondays are tried. For each one the forecaster only
    sees loans before that Monday and predicts the next ``horizon`` weeks.
      genre level: error per genre per week
      book level:  error per book on its total over the horizon (much noisier)
    Baselines need no AI: "4-week average" repeats the average of the previous 4
    weeks; "last 4 weeks repeated" assumes the next 4 weeks equal the last 4.
    WAPE = total absolute error / total real loans (lower is better).
    """
    as_of = pd.Timestamp(as_of).normalize()
    this_monday = as_of - pd.Timedelta(days=as_of.weekday())
    genres = sorted(books["genre"].unique())
    book_ids = books["book_id"].tolist()

    tagged = loans[["book_id", "borrowed_on"]].merge(books[["book_id", "genre"]], on="book_id")
    tagged["week"] = tagged["borrowed_on"].dt.to_period("W-SUN").dt.start_time
    by_genre = tagged.groupby(["week", "genre"]).size().unstack(fill_value=0).reindex(columns=genres, fill_value=0)
    by_book = tagged.groupby(["week", "book_id"]).size().unstack(fill_value=0).reindex(columns=book_ids, fill_value=0)

    errors = {"genre": {"model": [], "4-week average": []}, "book": {"model": [], "last 4 weeks repeated": []}}
    totals = {"genre": 0.0, "book": 0.0}
    for k in range(origins):
        origin = this_monday - pd.Timedelta(weeks=horizon + k)
        future = [origin + pd.Timedelta(weeks=i) for i in range(horizon)]
        past = [origin - pd.Timedelta(weeks=i) for i in range(1, 5)]
        forecaster = DemandForecaster(books, loans[loans["borrowed_on"] < origin], origin)

        g_pred = forecaster.forecast(by="genre", horizon_weeks=horizon)
        g_pred["week_start"] = pd.to_datetime(g_pred["week_start"])
        g_pred = (g_pred.pivot(index="week_start", columns="genre", values="predicted_loans")
                  .reindex(index=future, columns=genres, fill_value=0.0).to_numpy())
        g_actual = by_genre.reindex(future, fill_value=0).to_numpy(dtype=float)
        g_base = np.tile(by_genre.reindex(past, fill_value=0).mean().to_numpy(dtype=float), (horizon, 1))
        errors["genre"]["model"].append(np.abs(g_pred - g_actual).ravel())
        errors["genre"]["4-week average"].append(np.abs(g_base - g_actual).ravel())
        totals["genre"] += g_actual.sum()

        b_pred = (forecaster.forecast(by="book", horizon_weeks=horizon).groupby("book_id")["predicted_loans"]
                  .sum().reindex(book_ids, fill_value=0.0).to_numpy())
        b_actual = by_book.reindex(future, fill_value=0).sum().to_numpy(dtype=float)
        b_base = by_book.reindex(past, fill_value=0).sum().to_numpy(dtype=float)
        errors["book"]["model"].append(np.abs(b_pred - b_actual))
        errors["book"]["last 4 weeks repeated"].append(np.abs(b_base - b_actual))
        totals["book"] += b_actual.sum()

    result = {"origins": origins, "horizon": horizon}
    for level, methods in errors.items():
        result[level] = {}
        for name, errs in methods.items():
            flat = np.concatenate(errs)
            result[level][name] = {"mae": float(flat.mean()), "wape": float(flat.sum() / totals[level])}
    return result


# ---------------------------------------------------------------- chatbot
# Two sets of sentences the chatbot was NOT trained on. The DEV set was used while
# improving the training data. The TEST set was only looked at once, at the end,
# so its score is an honest estimate. Format: (sentence, correct intent).
CHATBOT_DEV = [
    ("do you have Deep Learning in stock", "availability"), ("is Atomic Habits on the shelf", "availability"),
    ("any copies of Sapiens left", "availability"), ("can i take out The Hobbit today", "availability"),
    ("which books must i return before friday", "due_soon"), ("what loans expire soon", "due_soon"),
    ("show what is due in the coming days", "due_soon"), ("anything due next week", "due_soon"),
    ("list late books", "overdue"), ("who has not returned their books", "overdue"),
    ("show books past their due dates", "overdue"), ("do i have late books", "overdue"),
    ("what should i borrow next", "recommendations"), ("give me some book ideas", "recommendations"),
    ("recommend something based on what i read", "recommendations"), ("suggest a few titles for me", "recommendations"),
    ("find me something on data structures", "search"), ("books about leadership", "search"),
    ("i want to learn about soil", "search"), ("show books on constitutional law", "search"),
    ("what has chinua achebe written", "author_search"), ("books by orwell", "author_search"),
    ("show me titles by adichie that are available", "author_search"),
    ("which genres will be busy next month", "forecast"), ("predict exam period borrowing", "forecast"),
    ("what books will we run out of", "forecast"), ("which titles need extra copies", "forecast"),
    ("what is the fine for a late book", "policy"), ("how many days can i keep a book", "policy"),
    ("can i borrow more than five books", "policy"), ("do i get warnings before a due date", "policy"),
    ("what have i borrowed so far", "my_loans"), ("show my fines", "my_loans"),
    ("how many books am i holding", "my_loans"), ("what do i currently owe", "my_loans"),
    ("please extend my loan on this book", "renew"), ("renew sapiens for me", "renew"),
    ("i need another two weeks with it", "renew"), ("can you renew my book", "renew"),
    ("what is like atomic habits", "similar"), ("books similar to deep work", "similar"),
    ("find me more like this", "similar"),
    ("good evening", "smalltalk"), ("thanks a lot", "smalltalk"), ("see you", "smalltalk"),
    ("what can i ask", "help"), ("how do i use you", "help"), ("can you help me", "help"),
    ("what is the best football team", "unknown"), ("tell me about your day", "unknown"),
    ("how tall is mount kenya", "unknown"), ("calculate 15 percent of 200", "unknown"),
]

CHATBOT_TEST = [
    ("is Clean Code available?", "availability"), ("do you have 'Cosmos' right now", "availability"),
    ("are there copies of Grit?", "availability"), ("can i borrow The Alchemist", "availability"),
    ("is the book 'Mindset' in stock", "availability"),
    ("what do i have to return this week", "due_soon"), ("which loans fall due soon", "due_soon"),
    ("show me due dates for next week", "due_soon"), ("books i should bring back in the coming days", "due_soon"),
    ("are any of my books overdue", "overdue"), ("show late returns", "overdue"),
    ("which loans are overdue", "overdue"), ("list the overdue books please", "overdue"),
    ("who is late with books", "overdue"),
    ("recommend me something to read", "recommendations"), ("what book should i read after this", "recommendations"),
    ("give me book suggestions", "recommendations"), ("show recommendations for me", "recommendations"),
    ("i need a new book to read, any ideas", "recommendations"),
    ("find books about machine learning", "search"), ("do you have books on economics", "search"),
    ("i am looking for biology textbooks", "search"), ("books about the history of kenya", "search"),
    ("search for algorithms", "search"), ("anything about python programming", "search"),
    ("show me books by ngugi", "author_search"), ("what does adichie have in the library", "author_search"),
    ("which books by orwell can i borrow", "author_search"),
    ("forecast demand for next month", "forecast"), ("which books need more copies", "forecast"),
    ("predict borrowing by genre", "forecast"), ("what will be popular during exams", "forecast"),
    ("how long is the loan period", "policy"), ("what is the fine for being late", "policy"),
    ("how many books can i borrow", "policy"), ("can i renew a loan", "policy"),
    ("will i get reminders before the due date", "policy"), ("how much do i pay for overdue books", "policy"),
    ("show my current loans", "my_loans"), ("how much do i owe", "my_loans"),
    ("which books do i have out", "my_loans"),
    ("how many books have i borrowed right now", "my_loans"),
    ("renew my loan", "renew"), ("extend this book", "renew"), ("renew 'Clean Code'", "renew"),
    ("i would like to extend my loan", "renew"), ("please renew it", "renew"),
    ("books like Sapiens", "similar"), ("what else is similar to this book", "similar"),
    ("find similar books", "similar"), ("recommend books similar to Grit", "similar"),
    ("hey hi", "smalltalk"), ("thanks very much", "smalltalk"), ("bye for now", "smalltalk"),
    ("good day", "smalltalk"),
    ("what are your features", "help"), ("help me please", "help"), ("how can you assist me", "help"),
    ("what is the temperature in nairobi", "unknown"), ("who is the president of france", "unknown"),
    ("translate hello into french", "unknown"), ("recommend a good restaurant", "unknown"),
]


def evaluate_chatbot(chatbot, dataset=None) -> dict:
    """Intent accuracy on sentences the chatbot never trained on.

    strict accuracy    : right intent AND confident enough to answer (what a user experiences)
    accuracy if sure   : right intent among the answers it was confident about
    answered           : share of sentences it was confident enough to answer
    """
    from .config import CHATBOT_MIN_CONFIDENCE
    dataset = CHATBOT_TEST if dataset is None else dataset
    correct = answered = answered_correct = 0
    wrong = []
    per_intent = {}
    for sentence, expected in dataset:
        intent, confidence = chatbot.classify(sentence)
        sure = confidence >= CHATBOT_MIN_CONFIDENCE
        ok = sure and intent == expected
        correct += ok
        answered += sure
        answered_correct += sure and intent == expected
        stats = per_intent.setdefault(expected, [0, 0])
        stats[0] += ok
        stats[1] += 1
        if not ok:
            wrong.append((sentence, expected, intent, round(confidence, 2)))
    n = len(dataset)
    return {
        "sentences": n, "strict_accuracy": correct / n,
        "accuracy_if_sure": answered_correct / answered if answered else 0.0,
        "answered": answered / n, "wrong": wrong,
        "per_intent": {k: v[0] / v[1] for k, v in per_intent.items()},
    }