# Evaluation results

Generated 2026-11-21 06:49 on simulated data (100 books, 150 students, 7,267 loans, library date 2026-11-16, seed 42).

## 1. Search

Hit rate at 3: a correct book is in the top 3 results for **34 of 35 queries (97%)**. The queries include misspellings, descriptions with no title words, and author names.

Missed: “how to focus” (returned: Good to Great, A Mind for Numbers).

## 2. Recommender

Temporal split: the model sees loans only up to 60 days ago, then we check whether the books each student really borrowed afterwards are in their top 10. Hit rate = students with at least one correct book; recall = share of their future books found.

**established students** (147 students tested)

| Method | Hit rate @10 | Recall @10 |
|---|---|---|
| Hybrid (our model) | 81% | 33% |
| Content-based only | 71% | 30% |
| Collaborative only | 68% | 27% |
| Most popular (baseline) | 59% | 22% |
| Random (baseline) | 55% | 19% |

**new students (3 loans)** (148 students tested)

| Method | Hit rate @10 | Recall @10 |
|---|---|---|
| Hybrid (our model) | 66% | 20% |
| Content-based only | 53% | 14% |
| Collaborative only | 55% | 14% |
| Most popular (baseline) | 36% | 9% |
| Random (baseline) | 47% | 11% |

“Most popular” scores below random for new students. A likely reason is that in this simulated library students borrow mostly within their own interests, which one list for everyone ignores.

## 3. Demand forecast

Backtest: 6 past dates, each forecasting the next 4 weeks using only earlier data. WAPE is total error divided by total real loans (lower is better). Baselines need no AI: repeat the recent average.

| Level | Method | Average error (loans) | WAPE |
|---|---|---|---|
| genre | model | 2.55 | 26.5% |
| genre | 4-week average | 2.91 | 30.3% |
| book | model | 1.50 | 35.4% |
| book | last 4 weeks repeated | 1.97 | 46.5% |

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

## 5. Agent decisions on 2026-11-16 (dry run, nothing changed)

- 30 overdue loans, 11 due soon
- 8 overdue by more than 14 days (final notice)
- 24 books where the forecast says we need more copies
- 8 overdue copies of books that are fully out and in demand (priority chase)

## Limits of these results

- All data is **simulated**. The simulator builds in reading tastes, seasons and exam rushes, so these numbers show that each model can find such patterns, not how it would score on real students.
- The chatbot's two sentence sets are small. The TEST set was used to find two bugs after its first run, so its score is slightly optimistic.
- The chatbot can not reliably recognise requests that have nothing to do with the library; it only declines when it is unsure.