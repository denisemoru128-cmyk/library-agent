# Evaluation results

Generated 2026-11-25 10:26 on simulated data (100 books, 150 students, 7,033 loans, library date 2026-11-25, seed 42).

## 1. Search

Hit rate at 3: a correct book is in the top 3 results for **34 of 35 queries (97%)**. The queries include misspellings, descriptions with no title words, and author names.

Missed: “how to focus” (returned: Good to Great, A Mind for Numbers).

## 2. Recommender

Temporal split: the model sees loans only up to 60 days ago, then we check whether the books each student really borrowed afterwards are in their top 10. Hit rate = students with at least one correct book; recall = share of their future books found.

**established students** (146 students tested)

| Method | Hit rate @10 | Recall @10 |
|---|---|---|
| Hybrid (our model) | 68% | 28% |
| Content-based only | 64% | 26% |
| Collaborative only | 68% | 26% |
| Most popular (baseline) | 66% | 24% |
| Random (baseline) | 47% | 15% |

**new students (3 loans)** (147 students tested)

| Method | Hit rate @10 | Recall @10 |
|---|---|---|
| Hybrid (our model) | 66% | 20% |
| Content-based only | 60% | 16% |
| Collaborative only | 51% | 11% |
| Most popular (baseline) | 44% | 10% |
| Random (baseline) | 43% | 10% |

## 3. Demand forecast

Backtest: 6 past dates, each forecasting the next 4 weeks using only earlier data. WAPE is total error divided by total real loans (lower is better). Baselines need no AI: repeat the recent average.

| Level | Method | Average error (loans) | WAPE |
|---|---|---|---|
| genre | model | 2.71 | 28.6% |
| genre | 4-week average | 2.98 | 31.4% |
| book | model | 1.50 | 36.0% |
| book | last 4 weeks repeated | 1.85 | 44.4% |

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

## 5. Agent decisions on 2026-11-25 (dry run, nothing changed)

- 35 overdue loans, 18 due soon
- 6 overdue by more than 14 days (final notice)
- 18 books where the forecast says we need more copies
- 9 overdue copies of books that are fully out and in demand (priority chase)

## 6. How stable are these numbers? (5 simulated libraries)

The evaluation above is one simulated library. To see how much the numbers depend on that one library, the same evaluation was run on 5 libraries. Each has its own random seed (different students, tastes and loans) and its history ends on a different day (2026-11-25, 2026-11-14, 2026-11-03, 2026-10-23, 2026-10-12). The first library is the demo library.

| Measure | Lowest | Average | Highest |
|---|---|---|---|
| Recommender hit rate @10: hybrid (established students) | 68% | 69% | 75% |
| Recommender hit rate @10: content-based only | 60% | 63% | 67% |
| Recommender hit rate @10: most popular | 58% | 63% | 66% |
| Recommender hit rate @10: hybrid (new students) | 54% | 60% | 66% |
| Recommender hit rate @10: most popular (new students) | 26% | 33% | 44% |
| Forecast error by genre (WAPE): model | 23.5% | 26.1% | 28.6% |
| Forecast error by genre (WAPE): 4-week average | 29.0% | 38.9% | 53.6% |
| Forecast error by book (WAPE): model | 33.5% | 37.0% | 41.6% |
| Forecast error by book (WAPE): last 4 weeks repeated | 43.4% | 52.1% | 64.1% |

Search and the chatbot are not in this table: they are tested on the catalogue and on fixed sentence sets, not on loans, so every library would give the same score.

How often the model beat what it is compared with:

- Hybrid recommender beats “most popular” (established students): **5 of 5 libraries**.
- Hybrid recommender beats content-based only (established students): **5 of 5 libraries**.
- Hybrid recommender beats “most popular” (new students): **5 of 5 libraries**.
- Forecast beats the 4-week average (by genre): **5 of 5 libraries**.
- Forecast beats repeating the last 4 weeks (by book): **5 of 5 libraries**.

## Limits of these results

- All data is **simulated**. The simulator builds in reading tastes, seasons and exam rushes, so these numbers show that each model can find such patterns, not how it would score on real students.
- The chatbot's two sentence sets are small. The TEST set was used to find two bugs after its first run, so its score is slightly optimistic.
- The chatbot can not reliably recognise requests that have nothing to do with the library; it only declines when it is unsure.
