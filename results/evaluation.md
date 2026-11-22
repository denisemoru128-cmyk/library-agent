# Evaluation results

Generated 2026-11-21 12:24 on simulated data (100 books, 150 students, 7,051 loans, library date 2026-11-21, seed 42).

## 1. Search

Hit rate at 3: a correct book is in the top 3 results for **34 of 35 queries (97%)**. The queries include misspellings, descriptions with no title words, and author names.

Missed: “how to focus” (returned: Good to Great, A Mind for Numbers).

## 2. Recommender

Temporal split: the model sees loans only up to 60 days ago, then we check whether the books each student really borrowed afterwards are in their top 10. Hit rate = students with at least one correct book; recall = share of their future books found.

**established students** (146 students tested)

| Method | Hit rate @10 | Recall @10 |
|---|---|---|
| Hybrid (our model) | 70% | 29% |
| Content-based only | 72% | 30% |
| Collaborative only | 62% | 22% |
| Most popular (baseline) | 61% | 22% |
| Random (baseline) | 47% | 15% |

**new students (3 loans)** (144 students tested)

| Method | Hit rate @10 | Recall @10 |
|---|---|---|
| Hybrid (our model) | 65% | 17% |
| Content-based only | 60% | 17% |
| Collaborative only | 51% | 14% |
| Most popular (baseline) | 44% | 10% |
| Random (baseline) | 42% | 10% |

## 3. Demand forecast

Backtest: 6 past dates, each forecasting the next 4 weeks using only earlier data. WAPE is total error divided by total real loans (lower is better). Baselines need no AI: repeat the recent average.

| Level | Method | Average error (loans) | WAPE |
|---|---|---|---|
| genre | model | 2.35 | 24.2% |
| genre | 4-week average | 2.92 | 30.1% |
| book | model | 1.49 | 34.9% |
| book | last 4 weeks repeated | 2.06 | 48.2% |

Per-book forecasts are much noisier than per-genre ones, because one book has few loans per week.

## 4. Chatbot (intent recognition on sentences it never trained on)

Strict accuracy counts a sentence as right only if the intent is correct AND the bot was at least 45% sure. “Accuracy when sure” ignores the sentences it declined to answer.

| Set | Sentences | Strict accuracy | Accuracy when sure | Answered |
|---|---|---|---|---|
| DEV (used while tuning) | 52 | 92% | 100% | 92% |
| TEST (held out) | 62 | 92% | 95% | 97% |

Sentences the bot got wrong on the held-out set (expected intent, what it said, confidence):

- “what does adichie have in the library”: expected **author_search**, got *availability* (27%)
- “can i renew a loan”: expected **policy**, got *renew* (82%)
- “will i get reminders before the due date”: expected **policy**, got *policy* (39%)
- “translate hello into french”: expected **unknown**, got *smalltalk* (77%)
- “recommend a good restaurant”: expected **unknown**, got *recommendations* (71%)

“can i renew a loan” counts as wrong because the classifier alone says *renew*. The chat handler turns questions like this into the rules answer, so a student still gets the right reply and nothing is renewed.

## 5. Agent decisions on 2026-11-21 (dry run, nothing changed)

- 32 overdue loans, 11 due soon
- 6 overdue by more than 14 days (final notice)
- 22 books where the forecast says we need more copies
- 7 overdue copies of books that are fully out and in demand (priority chase)

## Limits of these results

- All data is **simulated**. The simulator builds in reading tastes, seasons and exam rushes, so these numbers show that each model can find such patterns, not how it would score on real students.
- The chatbot's two sentence sets are small. The TEST set was used to find two bugs after its first run, so its score is slightly optimistic.
- The chatbot can not reliably recognise requests that have nothing to do with the library; it only declines when it is unsure.