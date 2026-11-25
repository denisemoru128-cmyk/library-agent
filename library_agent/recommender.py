"""Hybrid book recommender.

Three kinds of evidence, combined:

1. CONTENT-BASED: "books that read like the ones you borrowed".
   Each book is a TF-IDF vector built from genre, title and description; cosine
   similarity between two vectors says how alike the books are.

2. COLLABORATIVE FILTERING: "students who borrowed what you borrowed also
   borrowed these". We build a student x book matrix of past loans and compute
   item-item cosine similarity: two books are similar when the same students
   borrow both. This finds links the text can not see (e.g. a statistics book
   and a machine-learning book that CS students read together).

3. PRIOR (cold start): a student with no history has nothing to compare with,
   so we use the interests they declared at registration plus what is popular.
   The prior's influence fades as the student's history grows: weight = 1/(1+n).

For a student, every book they borrowed votes for similar books. A vote is
stronger for recent loans (half-life of 180 days). The final score blends the
content and collaborative votes, then mixes in the prior.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import nlp
from .config import RECOMMEND_TOP_N

HALF_LIFE_DAYS = 180         # a loan from 180 days ago counts half as much as one today
POPULARITY_WINDOW_DAYS = 90  # "popular" means borrowed in the last 90 days
CF_WEIGHT = 0.6              # share of the collaborative score in the hybrid blend
METHODS = ("hybrid", "content", "collaborative", "popular")


def _scale(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rescale to 0..1 using only the candidate books, so different scores can be blended."""
    out = np.zeros_like(values, dtype=float)
    chosen = values[mask]
    if chosen.size and chosen.max() > chosen.min():
        out[mask] = (chosen - chosen.min()) / (chosen.max() - chosen.min())
    return out


class Recommender:
    def __init__(self, books: pd.DataFrame, students: pd.DataFrame, loans: pd.DataFrame, as_of):
        self.books = books.reset_index(drop=True)
        self.as_of = pd.Timestamp(as_of)
        self.students = students.set_index("student_id")
        self.loans = loans[loans["borrowed_on"] <= self.as_of]
        self.book_pos = {bid: i for i, bid in enumerate(self.books["book_id"])}
        n_books = len(self.books)

        # --- content similarity (book x book)
        docs = [nlp.normalize(f"{r.genre} {r.genre} {r.title} {r.description}") for r in self.books.itertuples()]
        vectors = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(docs)
        self.sim_content = cosine_similarity(vectors)
        np.fill_diagonal(self.sim_content, 0.0)  # a book is not "similar" to itself

        # --- collaborative similarity (book x book), from who borrowed what
        student_pos = {sid: i for i, sid in enumerate(self.students.index)}
        matrix = np.zeros((len(student_pos), n_books))
        for sid, bid in zip(self.loans["student_id"], self.loans["book_id"]):
            if sid in student_pos and bid in self.book_pos:
                matrix[student_pos[sid], self.book_pos[bid]] = 1.0
        self.sim_collab = cosine_similarity(matrix.T) if matrix.any() else np.zeros((n_books, n_books))
        np.fill_diagonal(self.sim_collab, 0.0)

        # used for explanations and "similar books": both kinds of evidence together
        self.sim_blend = 0.5 * self.sim_content + 0.5 * self.sim_collab

        # --- popularity in the recent past
        recent = self.loans[self.loans["borrowed_on"] > self.as_of - pd.Timedelta(days=POPULARITY_WINDOW_DAYS)]
        counts = recent["book_id"].value_counts()
        self.popularity = np.array([counts.get(bid, 0) for bid in self.books["book_id"]], dtype=float)

        # --- each student's history: {book position: recency weight}
        self._history_cache = {}
        latest = self.loans.groupby(["student_id", "book_id"])["borrowed_on"].max()
        for (sid, bid), when in latest.items():
            age_days = max((self.as_of - when).days, 0)
            self._history_cache.setdefault(sid, {})[self.book_pos[bid]] = 0.5 ** (age_days / HALF_LIFE_DAYS)

    # ------------------------------------------------------------ helpers
    def history(self, student_id: int) -> dict:
        return self._history_cache.get(student_id, {})

    def _prior(self, student_id: int) -> np.ndarray:
        """Declared interests x popularity: what a new student probably wants."""
        interests = set(filter(None, str(self.students.loc[student_id, "interests"]).split("|")))
        popularity = 0.05 + self.popularity / max(self.popularity.max(), 1.0)
        in_interest = self.books["genre"].isin(interests).to_numpy(dtype=float)
        return (0.25 + in_interest) * popularity

    def _reason(self, student_id: int, pos: int) -> str:
        hist = self.history(student_id)
        interests = set(filter(None, str(self.students.loc[student_id, "interests"]).split("|")))
        if hist:
            source = max(hist, key=lambda b: hist[b] * self.sim_blend[b, pos])
            if self.sim_blend[source, pos] > 0.02:
                return f"Because you borrowed “{self.books.at[source, 'title']}”"
        if self.books.at[pos, "genre"] in interests:
            return f"Popular in {self.books.at[pos, 'genre']}, one of your interests"
        return "Popular with students right now"

    # ------------------------------------------------------------ main API
    def recommend(self, student_id: int, top_n: int = RECOMMEND_TOP_N, method: str = "hybrid",
                  availability: pd.DataFrame | None = None, only_available: bool = False,
                  exclude_borrowed: bool = True, genre: str | None = None) -> pd.DataFrame:
        """Return the top books for one student, best first, each with a reason.
        ``genre`` keeps only books of that genre ("recommend me law books")."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        hist = self.history(student_id)
        n_books = len(self.books)

        candidates = np.ones(n_books, dtype=bool)
        if genre is not None:
            candidates &= (self.books["genre"] == genre).to_numpy()
        if exclude_borrowed:
            candidates[list(hist)] = False
        available = np.full(n_books, np.nan)
        if availability is not None:
            available = self.books[["book_id"]].merge(availability, on="book_id", how="left")["available"].to_numpy()
            if only_available:
                candidates &= available > 0

        weights = np.zeros(n_books)
        for pos, w in hist.items():
            weights[pos] = w
        content = _scale(weights @ self.sim_content, candidates)
        collab = _scale(weights @ self.sim_collab, candidates)

        if method == "popular":
            scores = _scale(self.popularity, candidates)
        elif method == "content":
            scores = content
        elif method == "collaborative":
            scores = collab
        else:  # hybrid
            prior_weight = 1.0 / (1.0 + len(hist))  # new students lean on interests and popularity
            evidence = CF_WEIGHT * collab + (1 - CF_WEIGHT) * content
            scores = (1 - prior_weight) * evidence + prior_weight * _scale(self._prior(student_id), candidates)

        scores = np.where(candidates, scores, -np.inf)
        order = [i for i in np.argsort(-scores, kind="stable")[:top_n] if np.isfinite(scores[i])]

        rows = [{
            "book_id": int(self.books.at[i, "book_id"]),
            "title": self.books.at[i, "title"],
            "author": self.books.at[i, "author"],
            "genre": self.books.at[i, "genre"],
            "score": float(scores[i]),
            "available": None if np.isnan(available[i]) else int(available[i]),
            "reason": self._reason(student_id, i),
        } for i in order]
        return pd.DataFrame(rows, columns=["book_id", "title", "author", "genre", "score", "available", "reason"])

    def similar_books(self, book_id: int, top_n: int = 5) -> pd.DataFrame:
        """'Readers who liked this also liked...' for a single book."""
        pos = self.book_pos[book_id]
        order = np.argsort(-self.sim_blend[pos], kind="stable")[:top_n]
        return pd.DataFrame({
            "book_id": self.books.loc[order, "book_id"].to_numpy(),
            "title": self.books.loc[order, "title"].to_numpy(),
            "genre": self.books.loc[order, "genre"].to_numpy(),
            "similarity": self.sim_blend[pos][order],
        })