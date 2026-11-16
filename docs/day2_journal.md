# Day 2 journal: baselines, a fair choice, one saved model

**Project:** prediction intervals for TTC subway delay length, with a coverage guarantee
**Date:** 15 November 2026
**Status:** done. Committed and pushed.

This is my record of Day 2: what I built, what I found, what I decided and
why. As with Day 1, each section is written so I can answer questions about
it without opening the code.

---

## 1. Day 2 in one minute

| | |
|---|---|
| Goal | Four baselines scored on held-out data, one chosen by a rule fixed in advance, one model saved for Day 3. |
| Output | `docs/day2_scores.csv`, `models/final.joblib` |
| Slices | fit 17,281 · valid 2,523 · calib 6,563 · test 7,448 (33,815 total, matches Day 1) |
| Chosen model | Gradient boosting, quantile loss, on `log1p(min_delay)` |
| Headline number | q90 pinball loss 21.2% below the global floor (1.774 against 2.251 minutes) |
| Honest caveat | Linear quantile regression was within 1.3%: a tie |
| Biggest surprise | Station by hour band was worse than one global number at q90 |
| Not yet fixed | Boosting's own q90 covers 88.5% of validation delays, not 90% |

The one sentence for Day 2: **the model sharpens the upper bound much more
than the median** (21.2% better at q90, 6.8% at q50), and that is the half
of the problem the project is about.

---

## 2. Concepts I need to be able to explain

### Four slices, not three

Day 1 set up train, calibration and test. Choosing between models needs
held-out data too, and neither calibration (reserved for the conformal
quantile) nor test (measured once) can be used. So the last three months of
training became a validation slice:

```
[ fit ................ | valid ] [ calibration ] [ test ]
2023-01-01        2025-01-01  2025-04-01      2026-01-01
```

Fit on `fit`, compare on `valid`, refit the winner on both. Calibration and
test stay closed.

### Pinball loss

The score for a predicted quantile. For quantile `q`, prediction `p` and
truth `y`:

- if the truth is above the prediction (too low): `q × (y − p)`
- if the truth is below the prediction (too high): `(1 − q) × (p − y)`

Worked example at q = 0.9 with a prediction of 10 minutes:

- truth 14: loss = 0.9 × 4 = **3.6**
- truth 6: loss = 0.1 × 4 = **0.4**

Missing low costs nine times more than missing high. That imbalance pushes
the best prediction up to the point where 90% of outcomes fall below it,
which is the 90th percentile. At q = 0.5 the two sides cost the same, and
the loss is half the absolute error. Lower is better; units are minutes.

### Quantile regression

Ordinary regression predicts the average. Quantile regression predicts a
chosen percentile, by minimising pinball loss instead of squared error. I
fit two: the 50th (a typical delay) and the 90th (the upper bound a rider
cares about).

### Why fitting on log1p is safe for quantiles

Delays are skewed (skew 21.5 raw, 1.82 after `log1p`). For an average,
fitting on a log scale and converting back gives a biased answer. For a
quantile it does not: `log1p` never changes the order of values, so the 90th
percentile of `log1p(Y)` is `log1p` of the 90th percentile of `Y`. Convert
back with `expm1` and the quantile is in minutes, exactly.

### The four baselines

| # | Baseline | Idea |
|---|---|---|
| 0 | Global quantiles | One median and one 90th percentile for everyone. The floor. |
| 1 | Group quantiles | Percentiles within station × hour band, falling back to line × hour band, then line, then global when a cell has fewer than 30 fit rows. |
| 2 | Linear quantile regression | One weight per category, fitted for each percentile. |
| 3 | Gradient boosting | Many small trees with quantile loss. Can learn combinations such as "this code at night". |

Boosting stops adding trees when an internal 10% hold-out, drawn at random
from the **fit** rows, stops improving. That is allowed: it only sets the
number of trees, and validation is still untouched.

### Features, and why the encoder sees only the fit slice

Line, hour band, day of week, delay code and station. Codes and stations
with fewer than 30 fit rows become `OTHER`, as does any value first seen
after the fit slice. Result: 61 code groups and 75 station groups.

Deciding which categories are common enough to keep is a choice made from
data. If it looked at calibration or test rows, those slices would shape the
model before they are used. So the lists are learned from fit and frozen.

`min_gap` is left out: it is the gap between trains *caused by* the delay.

### Why `share_below_q90` is not 0.90

It is the share of validation delays at or below the model's predicted 90th
percentile. A quantile model aims at 90% but has no guarantee on new data:
boosting got 88.5%. Day 3's conformal step uses the calibration slice to
measure that miss and widen the bound to correct it.

### Quantile crossing

The 50th and 90th percentile models are fitted separately, so nothing stops
the predicted 90th from coming out below the predicted 50th. That never
happened for any model (0.0 everywhere).

### Why refit on fit + valid

Validation existed only to choose. After the choice, those three months are
the most recent training data, and leaving them out makes the model worse
for no gain. The encoder is refitted on the same 19,804 rows.

### Temporal vs random split

A random split lets the model train on next month and be tested on last
month. If anything in the features tracks time, that flatters the score.
Mine carry no date, so I expected little effect, and found little: the
temporal result sat inside the range of five random splits. The temporal
split is still the right one, because only it matches real use.

---

## 3. Setup changes

Added to the end of `src/config.py`:

| Setting | Value | Why |
|---|---|---|
| `VALID_START` | `"2025-01-01"` | Start of the validation slice |
| `QUANTILES` | `(0.5, 0.9)` | The two percentiles every baseline predicts |
| `MIN_CATEGORY_ROWS` | `30` | Fit rows a code or station needs for its own category |
| `MODELS_DIR` | `PROJECT_ROOT / "models"` | Where fitted models go; ignored by git |

`models/` added to `.gitignore`: a model is rebuilt by script, so it does not
belong in the repo.

---

## 4. The pipeline, Day 2 additions

| Script | What it does |
|---|---|
| `data.py` | Loads the modelling population (delay filter, line filter, aliases, hour bands), tags each row fit, valid, calib or test, and builds features. Every script from now on imports it, so the population is defined once. Not run directly. |
| `08_baselines.py` | Fits the four baselines on fit, scores them on valid, applies the selection rule, writes `docs/day2_scores.csv`. Asserts nothing outside fit and valid is used. |
| `09_fit_final.py` | Refits the chosen model on fit + valid and saves it, with its encoder, to `models/final.joblib`. |
| `10_leakage.py` | Optional. Compares the temporal split with five random splits, inside the training period only. |

Row accounting printed by `data.py` on the real data:

```
rows in Parquet                          97,136
min_delay >= 1                           34,163  -62,973
line in YU, BD, SHP                      33,815  -348
fit        17,281  (51.1%)
valid       2,523  (7.5%)
calib       6,563  (19.4%)
test        7,448  (22.0%)
```

---

## 5. What went wrong, and what each one taught me

Much less than Day 1. That is itself worth recording.

**1. The plan had two mistakes, caught before any code.**
It said baselines would be "scored on a temporal test set", which would have
spent the test set on Day 2. And it proposed a station-by-hour median, which
most stations cannot support.
*Lesson: reread a plan against the project's rules before following it,
even if I wrote it myself.*

**2. I logged the selection rule after the run, not before.**
The runbook said to write it first. The rule was already fixed in the script
(`SELECTION_RULE`, printed before the scores), so the choice could not
drift, but the log entry is not the evidence of that.
*Lesson: pre-register in the place that cannot quietly change afterwards.
Here that was the code.*

**3. The test data for the Day 2 scripts guessed parts of my schema wrong.**
The stand-in had a `year` column (mine has none), a numeric `month` (mine is
text) and no `id` column. Printing the real dtypes and null counts before
Step 4 showed none of those columns is read today, so nothing broke. It also
showed 279 missing `line` values in the full file, not 2: the other 277 are
zero-delay rows, removed by the delay filter before the line filter runs.
*Lesson: same as Day 1. Check the real schema before trusting a test,
however close the stand-in looks.*

---

## 6. Results

### Baselines on the validation slice (minutes; lower pinball is better)

| Model | pinball q50 | pinball q90 | share below q90 | q90 below q50 |
|---|---|---|---|---|
| 0 global quantiles | 2.131 | 2.251 | 91.1% | 0.0 |
| 1 group quantiles | 2.098 | 2.316 | 89.5% | 0.0 |
| 2 linear quantile | 1.995 | 1.797 | 89.0% | 0.0 |
| 3 gradient boosting | 1.987 | **1.774** | 88.5% | 0.0 |

How to read it:

- **Boosting won on the rule, but only just.** 1.3% ahead of linear on 2,523
  rows is a tie. I report that it won, and that the margin is not one I
  would defend.
- **The gain is in the tail.** Against the floor: 6.8% better at q50,
  21.2% better at q90. My reading, not yet tested: the delay code mostly
  separates long incidents from routine ones.
- **Station by hour band lost to the floor at q90** (2.316 against 2.251),
  even though 82% of validation rows were answered by a station × hour-band
  cell. A tail quantile from as few as 30 rows is noise. This backs up the
  Day 1 decision not to use station as a conformal group.
- **Every learned model undercovers on new data** (88.5% to 89.5%); global
  quantiles overcover (91.1%).

### Final model

Refitted on 19,804 rows. In-sample share below q90: 90.6%, against 88.5% on
validation. The in-sample figure is optimistic by design and is not a
result; the gap is why Day 3 calibrates on held-out rows.

### Leakage check (median model, 2,523 held-out rows each)

| Split | q50 pinball loss |
|---|---|
| Temporal (fit → valid) | 1.987 |
| Random, mean of 5 seeds | 1.920 (range 1.806 to 2.120) |

The temporal result sits inside the random range: little leakage with
these features. The 3.4% average gap could be slight leakage or a slightly
harder three months, and five seeds cannot tell which. The spread across
random sets (0.31 minutes) also gives a rough sense of how noisy any
2,523-row comparison is, which supports calling the Step 6 margin a tie.

---

## 7. Decisions, each with its reason

| Decision | Reason |
|---|---|
| Validation slice Jan–Mar 2025, cut from training | Calibration is reserved for the conformal quantile; test is measured once. |
| Selection rule: lowest q90 pinball on validation | The deliverable is an upper bound. Fixed in code before the run. |
| Predict 0.5 and 0.9 | 0.9 is the deliverable; 0.5 keeps both Day 3 conformity scores possible. |
| 30 fit rows for a category, else `OTHER` | Rare categories are noise; unseen 2026 codes need somewhere to go. |
| Encoder learned on fit only | Calibration and test must not shape the features. |
| No `min_gap` | It is a consequence of the delay. |
| Fit on `log1p`, convert with `expm1` | Skew 21.5; quantiles survive the transform exactly. |
| Group baseline falls back at 30 rows | Most station × hour-band cells are too thin on their own. |
| Refit on fit + valid | Validation was only for choosing. |
| Keep the temporal split despite a small leakage gap | Only it matches how the model would be used. |

---

## 8. My prediction for Day 3 (written before any conformal code)

> FILL IN — commit to numbers before Day 3. Boosting's own q90 covered
> 88.5% of validation delays.
>
> - Raw coverage of the q90 on the calibration slice, before correction:
> - How much the conformal step will widen the bound (a little or a lot,
>   and why):
> - Which group I expect to undercover first on Day 4:

---

## 9. Questions I should be able to answer quickly

**Why not choose the model on the calibration set?**
Day 3 computes the conformal quantile from calibration errors. If those rows
had also chosen the model, they would not be fresh data for it, and the
guarantee would rest on reused data.

**What does pinball loss measure?**
How well a predicted percentile does its job. At q = 0.9, missing low costs
nine times more than missing high. At q = 0.5 it is half the absolute error.

**Why fit on the log scale?**
The delays are heavily skewed, and quantiles pass unchanged through any
increasing transform, so nothing is lost converting back.

**Boosting and linear were tied. Why use boosting?**
Because the rule, fixed before the run, said lowest q90 loss, and boosting
had it. I report the tie rather than claiming boosting is better. Conformal
works with either; the model only decides how wide the intervals are.

**Why did station by hour band do worse than one global number?**
A 90th percentile estimated from 30 rows is noisy, and that noise cost more
than station detail gained.

**Boosting's q90 covers 88.5%, not 90%. Is the model broken?**
No. A quantile model aims at 90% with no guarantee on new data. That gap is
exactly what conformal prediction corrects on Day 3.

**Did a random split flatter the model?**
Slightly on average (3.4%), but the temporal score sat inside the random
range. My features carry no date, which closes the main leakage route.

**Why refit on fit + valid?**
Validation was only for choosing. Afterwards it is the most recent training
data.

**Why leave out `min_gap`?**
It is caused by the delay, so it is not known before it.

**Why do codes and stations need 30 fit rows?**
Below that, a category's own estimate is noise. Rare and new values pool
into `OTHER`, which also handles codes missing from the official key.

---

## 10. Commands I run to check everything

```powershell
python src/07_verify.py                              # expect "All 31 checks passed"
python -c "import sys; sys.path.insert(0,'src'); from data import load_population; load_population()"
                                                     # expect fit 17,281 · valid 2,523 · calib 6,563 · test 7,448
python src/08_baselines.py                           # the four scores and the chosen model
python src/09_fit_final.py "3 gradient boosting"     # writes models/final.joblib
python src/10_leakage.py                             # optional
```

---

## 11. Open items

- [ ] Fill in the Day 2 reading lines in `log.md` (and Day 1's, still blank)
- [ ] Write the Day 3 prediction (section 8) before any conformal code
- [ ] Check the AI assistance note in `log.md` is accurate
- [ ] Day 3 starts with: load `models/final.joblib`, predict on the calibration
      slice, and choose the conformity score: a one-sided upper bound from the
      q90 prediction, or a two-sided interval around the median
- [ ] Optional: the share of rows whose code fell into `OTHER`, only if
      Day 4 shows that group undercovering

---

## AI acknowledgement

*(to be written by me)*
