# Project log

## Day 1

Goal for the day: raw files in, one clean Parquet out, the prediction target
written down. Nothing modelled.

### Decisions

**Date range: 2023-01-01 onward.** The TTC system changed substantially over
the twelve years the portal covers — pandemic ridership collapse, then the
Line 3 Scarborough RT closure in July 2023. Training across different
operating regimes makes a model worse, not better, and conformal prediction
assumes calibration and test data are exchangeable, which a 2014-to-2026 span
plainly violates. 2023 onward gives 45 months of a reasonably stable system.
Evidence: `docs/figures/q3_incidents_per_month.png` shows monthly incident
counts between 1,736 and 2,715 with no gaps or step changes.

**Split dates: TRAIN_END 2025-04-01, CALIB_END 2026-01-01.** Chronological,
never random. Rows before TRAIN_END train the model; rows from TRAIN_END to
CALIB_END are the calibration set the conformal quantile is computed from;
everything after CALIB_END is the test set, touched once at the end. Both
dates were rounded to month starts after picking the nearest 60/80% row
quantiles, so the slices are readable in the write-up.

Slice counts over all 34,163 delayed rows (before the line filter), verified
disjoint, exhaustive and strictly chronological by `src/07_verify.py`:

| Slice | Rows | Share | Range |
|---|---|---|---|
| Train | 20,146 | 59.0% | up to 2025-03-31 23:43 |
| Calibration | 6,565 | 19.2% | 2025-04-01 02:59 to 2025-12-31 23:21 |
| Test | 7,452 | 21.8% | from 2026-01-01 00:12 |

**Modelling population: lines YU, BD and SHP only.** The `line` column holds
13 distinct values, not 4. SRT (Line 3) has 339 delayed rows, the last on
2023-07-24, so it sits entirely inside the training period and contributes
nothing to calibration or test; it is dropped rather than left to leak a
closed line's behaviour into the model. `YUS` (3 rows) and `BD LINE 2`
(1 row) are alternate spellings and are remapped. Seven labels are bus or
streetcar routes with one row each — `62 MORTIMER`, `506 CARLTON`,
`504 KING`, `20 CLIFFSIDE`, `109 RANEE`, `76 ROYAL YORK SOUTH`,
`77 SWANSEA` — wrong mode of transport, dropped.

**Station will not be a conformal group.** 610 distinct station strings for
roughly 70 real stations. In the calibration slice only 21 of 277 stations
reach 100 rows, and the 256 that do not hold 55% of the slice. A
per-station quantile would be noise. Day 4 groups by line, hour band and
delay-code category instead; stations stay as a descriptive breakdown with
the largest ones plus an OTHER bucket.

**Hour bands, not raw hours.** Hours 2, 3 and 4 hold 141, 14 and 21 delayed
rows in the whole dataset. Six bands (overnight 00–04, early 05–06, AM peak
07–09, midday 10–15, PM peak 16–18, evening 19–23) each clear 440 rows in the
calibration slice. Stored in `config.py` so Day 2 and Day 4 cannot disagree.

### Numbers

| | |
|---|---|
| Rows in Parquet | 97,136 |
| Date range | 2023-01-01 01:20 to 2026-09-30 23:59 (45 months) |
| Share with `min_delay` >= 1 | 35.2% (34,163 rows) |
| Delay distribution | median 5, p90 14, p99 46, max 716 minutes |
| Skew | 21.5 raw, 1.82 after `log1p` |
| Incidents per month | 1,736 to 2,715, no thin months |
| Distinct delay codes | 197 in the data, 178 among delayed rows |
| Rush vs off-peak spread | IQR 4.0 both; p90 13 rush, 15 off-peak; ratio 1.00x |
| Stations | 610 distinct; 76 cover 94.5% of delayed rows |

The number that justifies the project: **median 5 minutes, 90th percentile
14**. A rider told "about 5 minutes" faces 14 or more one time in ten. A point
forecast hides that; an interval does not.

**The implausible-delay ceiling is barely doing anything.**
`MAX_PLAUSIBLE_DELAY_MIN` is 720 (12 hours) and it removed exactly 3 rows of
97,331. The largest surviving delay is 716 minutes, four minutes under the
cutoff. So the ceiling is not quietly discarding a tail of real long delays,
but it is also not a meaningful filter — it is a guard against data-entry
errors that this extract happens not to contain. Worth saying plainly rather
than implying the threshold was tuned.

**Data-quality finding (Q5).** The official key `Code_Descriptions.csv`
lists 140 codes; the data uses 197. Measured over every delayed row, 54 codes
are absent from the key, covering 778 rows (2.3%) and 3.2% of delay minutes.
That figure mostly measures the Line 3 SRT: dropping the 339 SRT rows removes
338 of those 778 rows, because the SRT's equipment codes (`ER*`, `SR*`) are
not in this key at all. They are listed side by side with the subway codes in
a separate file, `ttc-subway-delay-codes.xlsx`.

The number that matters for modelling is the one after SRT is excluded: in the
33,815-row modelling population, 5 of 129 codes are absent from the key,
covering 1.3% of rows. `PUTO` is the largest (281 rows, 3,005 minutes across
all lines). The five, as printed by section 4 of `07_verify.py`:

| Code | Rows in population |
|---|---|
| `PUTO` | 281 |
| `SUL` | 121 |
| `MUFL` | 17 |
| `MUBIO` | 17 |
| `TUAV` | 1 |

Any code-based feature therefore needs an explicit "unknown code" category
rather than an assumption that the key is complete. This is quality-check
material for Part 4.

### Target paragraph

The population is subway delay incidents logged by the TTC on Line 1 (YU),
Line 2 (BD) and Line 4 (SHP) between 1 January 2023 and 30 September 2026,
restricted to incidents where the recorded delay was at least one minute —
33,815 of the 97,136 logged records (34,163 delayed rows before the line
filter). The target is `min_delay`, the minutes
of delay to service caused by that incident, and the deliverable is a 90%
prediction interval for it rather than a point estimate. At prediction time I
assume the station, line, timestamp, day of week and the TTC's delay reason
code are known. I am not predicting whether a delay will occur, only how long
one lasts given that one has been logged. Because the reason code is assigned
by TTC staff rather than being visible to a rider at the moment a train
stops, this is a retrospective analysis rather than a deployable rider-facing
tool; a variant without the code feature would be the deployable version, and
comparing the two is a stretch goal, not a commitment. I am also not claiming
conditional coverage. The conformal guarantee is marginal — 90% averaged over
everything — and Day 4 exists to find where it fails on subgroups.

### Prediction for Day 4

Marginal coverage will land within a point or two of 90% on the test set.
I expect undercoverage in groups dominated by long-tail delay codes — `MUPR1`
has a median of 72.5 minutes against 5 overall, and `MUPLB` 25 — because one
global interval width cannot stretch to cover an hour-long shutdown while
staying narrow enough to be useful for a five-minute delay. I expect hour of
day to be a weak grouping: the pooled rush-hour and off-peak IQRs came out
identical at 4 minutes. If no group undercovers once the coverage estimates
are bootstrapped, I will report that as a negative result rather than slice
until something breaks.

### Surprised me

**The rush-hour hypothesis did not survive contact with the data.** The plan
predicted that delays at rush hour would be both longer and more variable, so
that one global interval width could not serve peak and off-peak alike. They
are not. Pooled across all delayed rows the interquartile range is 4 minutes
at both, and the 90th percentile is *lower* at rush hour (13 against 15). The
hourly table shows the morning peak is the tightest part of the day
(IQR 3 minutes at 07:00–09:00) and the widest hours are late evening and
after midnight. The first version of the check also averaged the 24 hourly
IQRs, which gave hour 3 with its 14 rows the same weight as hour 8 with
2,224; pooling the raw rows instead changed the answer. Both the wrong
hypothesis and the wrong weighting are worth keeping in the write-up.

**Two files on the portal are mislabelled.**
`TTC_Subway_Delay_Data_since_2025.xlsx` is CSV text with an `.xlsx` name, and
so is `Code_Descriptions.xlsx`. The first build therefore stopped at
31 December 2024 and lost 47,915 rows — every record from 2025 on — while
still printing a plausible-looking total. The build now checks what a file
contains rather than trusting its extension.

**Mixed date formats break silently when files are stacked.** pandas infers
one date format from the first rows it sees and turns everything in another
format into `NaT`. Stacking the files first and parsing afterwards destroyed
thousands of timestamps without raising anything. Each file's `event_time` is
now built before the files are concatenated.

**The schema check was in the wrong place.** It ran on the concatenated
table, where a column missing from one file is supplied by the others, so an
entire year could be dropped without the check noticing. It now runs per file
and names the file that failed. Verified by renaming a column in a raw file
and watching the build stop.

**`line` and `station` are free text, not categories.** `line` contains bus
and streetcar routes; `station` contains entrance and platform annotations
(`VMC STATION PLATFORM 1`, `KENNEDY BD STATION [PL`) and even segments
between stations (`UNION STATION TO ST AN`). Several values look cut off at a
fixed width.

## Day 2 — 15 November 2026

Goal for the day: four baselines scored on held-out data, one chosen by a rule
fixed in advance, and one saved model for Day 3. Calibration and test not used
by anything.

### Two corrections to the plan

**Correction 1: baselines are not scored on the test set.** The plan said
"scored on a temporal test set". That breaks the project's own rule that the
test set is measured once, at the end. The calibration set cannot be used
either, because Day 3 needs it untouched for the conformal quantile. So
choosing between baselines needs a fourth slice, cut from the end of the
training period:

```
[ fit ................ | valid ] [ calibration ] [ test ]
2023-01-01        2025-01-01  2025-04-01      2026-01-01
```

**Correction 2: a plain station-by-hour median does not work.** Only a few
dozen stations have enough rows. The baseline uses a fallback instead:
station and hour band if that cell has at least 30 fit rows, otherwise line
and hour band, otherwise line, otherwise the overall value.

### Decisions

**`VALID_START = 2025-01-01`.** Validation is January to March 2025, the last
three months of training: 2,523 rows. Fit is everything before it: 17,281
rows. Together they are the 19,804 training rows `07_verify.py` reports for
the modelling population, so the population is defined the same way in both
places. Calibration (6,563) and test (7,448) also match.

**Every baseline predicts two quantiles, 0.5 and 0.9.** The 0.9 is the
deliverable, an upper bound. The 0.5 is kept so Day 3 can choose between a
one-sided bound and an interval around the median.

**Codes and stations need 30 fit rows for their own category.** Rarer ones,
and any value first seen after the fit slice, pool into `OTHER`. The category
lists are learned from the fit slice only, so calibration and test cannot
influence the features. This is also how codes missing from the official key
are handled: they are just more categories, or `OTHER`.

**`min_gap` is not a feature.** It is the gap between trains caused by the
delay, so using it would predict the delay from its own consequence.

**Models 2 and 3 fit on `log1p(min_delay)`.** Delays are heavily skewed
(skew 21.5 raw, 1.82 after `log1p`). Quantiles survive any increasing
transform, so converting back with `expm1` gives the quantile in minutes with
no bias.

### Selection rule (Step 5)

I will choose the baseline with the lowest pinball loss at the 90th
percentile on the validation slice. Why the 90th rather than the 50th: the
deliverable is "90% chance it clears within X", which is an upper bound.

The rule was fixed in `08_baselines.py` (`SELECTION_RULE`), which prints it
before any score. I wrote it into this log after the run, so the script, not
the log, is the evidence that it came first.


### Baselines (Step 6)

**Selection rule, fixed in `08_baselines.py` before the run:** lowest pinball
loss at q=0.9 on the validation slice (Jan–Mar 2025, 2,523 rows). Models were
fitted on the fit slice (17,281 rows). Calibration and test were not used.

| Model | pinball q50 | pinball q90 | share below q90 | q90 below q50 |
|---|---|---|---|---|
| 0 global quantiles | 2.131 | 2.251 | 91.1% | 0.0 |
| 1 group quantiles | 2.098 | 2.316 | 89.5% | 0.0 |
| 2 linear quantile | 1.995 | 1.797 | 89.0% | 0.0 |
| 3 gradient boosting | 1.987 | 1.774 | 88.5% | 0.0 |

**Chosen by the rule: gradient boosting.** Its q90 pinball loss is 21.2% below
the global floor. The margin over linear quantile regression is 1.3%
(1.774 against 1.797), which on 2,523 validation rows is too small to call one
better than the other. Boosting wins on the rule I wrote down, not by a margin
I would defend.

**The models improve the upper bound much more than the median.** Against the
floor, boosting cuts q50 loss by 6.8% but q90 loss by 21.2%. My reading: the
delay code mostly separates the long incidents from the routine ones, so it
sharpens the tail more than the typical case. That fits the Day 1 finding
that code, not hour, is where the spread lives.

**The station-by-hour baseline is worse than one global number at q90**
(2.316 against 2.251). It answered 82% of validation rows from a
station × hour-band cell, so this is not a fallback problem. A 90th percentile
estimated from as few as 30 rows is noisy, and that noise costs more than the
station detail gains. Same conclusion as Day 1's decision not to use station
as a conformal group, reached from a different direction.

**No model hits 90% on new data.** Boosting's own q90 covers 88.5% of
validation delays, and the models that learned more undercover more. Global
quantiles overshoot (91.1%). None of these has a coverage guarantee; that gap
is what Day 3's conformal step corrects, using the calibration slice.

**No quantile crossing.** The predicted q90 was never below the predicted q50.

Features: 61 code groups and 75 station groups, learned from the fit slice;
rarer codes and stations, and anything first seen later, pool into OTHER.

### Final model (Step 7)

Refitted gradient boosting on fit + valid: 19,804 rows, all of training.
Saved to `models/final.joblib` with its encoder, which was also refitted on
those 19,804 rows. Calibration and test still untouched.

The validation months were held back only to choose a model. Once the choice
was made they were just the most recent training data, so leaving them out
would have made the model worse for no gain.

In-sample share below q90 is 90.6%, against 88.5% on validation in Step 6.
The in-sample figure is measured on rows the model was fitted on, so it is
optimistic by design and is not a result. The 2-point gap between the two is
a small example of why Day 3 calibrates on held-out rows rather than training
rows.

### Leakage check (Step 8)

Same boosting median model, two ways of holding out 2,523 training rows:

| Split | q50 pinball loss |
|---|---|
| Temporal (fit → valid, Jan–Mar 2025) | 1.987 |
| Random, mean of 5 seeds | 1.920 (range 1.806 to 2.120) |

**The temporal result sits inside the random range, so I found little
leakage with these features.** The random split averages 3.4% lower, but
five random held-out sets spread from 1.806 to 2.120, a range far wider than
that gap. None of my features carries the date (no year, month or trend),
which closes the main route for the future to leak into the past. I am
keeping the temporal split anyway: it is the only one that matches how the
model would be used, and a small gap here does not prove there is none
elsewhere.

**A side result about Step 6.** Swapping one random set of 2,523 held-out rows
for another moved the median loss by up to 0.31 minutes. That is a rough
measure of how noisy a 2,523-row comparison is. It supports calling the 1.3%
gap between boosting and linear quantile regression a tie. This check covers
only the median and five seeds, so I treat it as a sense of scale, not a
formal test.

### Surprised me

**Station detail made the upper bound worse, not better.** The obvious
baseline, a 90th percentile per station and hour band, lost to one global
number. More detail is only better when each cell has enough rows to estimate
a tail quantile.

**A linear model did as well as boosting.** One weight per category came
within 1.3% of gradient boosting. With these features there seem to be few
interactions, such as "this code at night", worth learning.

**Nothing broke today.** Every count matched Day 1 on the first run, after
Day 1 needed three attempts at its own acceptance test.

## Day 3 — 16 November 2026

Goal for the day: turn Day 2's saved model into 90% prediction intervals
with split conformal prediction, calibrate on the calibration slice, and
measure coverage on the test set exactly once.

### Read
- Angelopoulos and Bates, split conformal procedure: the new score is
  equally likely to land in any of the n + 1 positions among the calibration
  scores, so taking the k-th smallest with k = ceil((n + 1) × 0.9) gives at
  least 90% coverage; using n would undercover slightly.
- Same paper, conformalized quantile regression: start from a model that
  already predicts quantiles and use calibration only to correct how far
  they miss, so the interval keeps the model's adaptive width. My `"upper"`
  score is the one-sided version.
- My Day 1 journal, section 2: today measures marginal coverage only, and
  relies on 2025 and 2026 being exchangeable, which holds only approximately.


### Conformity score (Step 1, chosen before any code ran)

**Correction to the plan: not absolute residuals in minutes.** The plan said
plain split conformal on absolute residuals would be enough. That gives the
prediction plus or minus one fixed number of minutes. My delays are heavily
skewed (median 5, 90th percentile 14, maximum 716), so a symmetric interval
is the wrong shape: its lower end falls below 1 minute, which no delay in my
population can be, and it gives a five-minute delay and an hour-long
shutdown the same width.

**Chosen: `CONFORMAL_SCORE = "upper"`.** The score is the actual delay minus
the model's predicted 90th percentile, in minutes. The interval runs from
1 minute up to the predicted 90th percentile plus `q_hat`, and reads as
"90% chance it clears within X minutes". I chose it because the project's
question is how long until a delay clears, which is an upper bound, and
because Day 2 showed the model helps most at the 90th percentile (21.2%
below the floor, against 6.8% for the median). The bound is built on the
part of the model that works best. Fixing the lower end at 1 minute costs
nothing, since every delay in the population is at least 1 minute.

The alternative, `"two_sided"` (log-scale distance from the predicted
median), carries the same guarantee with a different shape. I fixed the
choice before running anything, for the same reason the Day 2 selection
rule was fixed first: so it cannot drift toward whichever result looks best.

### Settings (Step 2)

Added the Day 3 block to the end of `src/config.py`: `CONFORMAL_SCORE =
"upper"` and the three output paths (`DAY3_CALIBRATION`,
`DAY3_TEST_RESULTS`, `DAY3_TEST_ROWS`). The score lives in `config.py`, as
the split dates do, so no script can quietly use a different one.
Checked: `config.CONFORMAL_SCORE` prints `upper`, and `07_verify.py` still
passes all 31 checks, so the edit did not disturb Day 1.

### Scripts (Step 3)

Copied the four Day 3 scripts into `src/`: `conformal.py` (the method),
`11_calibrate.py` (computes `q_hat` from calibration, never reads a test
delay), `12_test_once.py` (measures the test set once, then refuses to run
again) and `13_plot.py` (draws the figure from saved results). Nothing has
been run yet. 

### Conformal method, by hand (Step 4)

Before running anything I worked three cases on paper.

- Nine scores, 1 to 9: n + 1 = 10, 10 × 0.9 = 9, so k = 9 and `q_hat` is the
  9th smallest, which is 9. A new score lands in one of 10 positions and is
  covered in 9 of them: 9/10. I first wrote 9 out of 9, which forgets that
  the new score takes a position in the line too.
- Nineteen scores, 1 to 19: n + 1 = 20, 20 × 0.9 = 18, so k = 18 and
  `q_hat` = 18. Chance of coverage 18/20.
- Five scores, 1 to 5: (5 + 1) × 0.9 = 5.4, so k = 6, more than the 5 scores
  I have. Using the biggest score would cover only 5/6 = 83%, so no finite
  bound can promise 90% and the honest `q_hat` is infinite. This never
  happens with my 6,563 calibration rows (k = 5,908).

**Why n + 1.** I line up the n calibration scores and the one new score.
If the rows are exchangeable, the new score is equally likely to land in any
of the n + 1 positions. It is covered when it lands in the bottom k, so the
chance is k / (n + 1), and k = ceil((n + 1) × 0.9) makes that at least 0.9.
Using n instead would undercover: with n = 10, k = 9 gives 9/11 = 82%. The
argument never uses the model, only that the rows are exchangeable. That is
why a worse model gives wider intervals, not invalid ones.

Self-test (`conformal.py`): all four checks passed, including a simulation
of 20,000 made-up worlds whose coverage matched k / (n + 1) = 91/101.

### Prediction for the calibration step (written 16 November, before running 11_calibrate.py)

- **The model's own 90th percentile will cover about 88.5% of calibration
  delays** (I'd accept 87.5% to 89.5%). Validation gave 88.5% and the
  in-sample figure was 90.6%. Calibration is unseen data like validation,
  the three months after training, so I expect it to behave like validation,
  not like the training rows.
- **The conformal step will widen the bound by a little: `q_hat` positive,
  around 1 minute, under 2.** It only has to catch the extra ~1.5% of delays
  just above the predicted bound, and delays are whole minutes, so a small
  shift covers them. A `q_hat` of several minutes would mean calibration
  differs from validation more than I think.
- **The no-model bound will be about 14 minutes**, the population's 90th
  percentile. Most delays should get a tighter bound from the model,
  because the delay code separates routine stops from long incidents.

### Calibration (Step 5, `11_calibrate.py`)

The self-test passed. The model loaded as `3 gradient boosting` fitted on
19,804 rows, against 6,563 calibration rows (2025-04-01 to 2025-12-31).

- **Before conformal:** the model's own q90 covered **89.8%** of calibration
  delays (validation gave 88.5%).
- **`n` = 6,563, `k` = 5,908**, level 0.9002. **`q_hat` = 0.0835 minutes**:
  conformal raises every predicted q90 by under a tenth of a minute.
  Sections 3 and 4 agree: coverage slightly under 90%, correction positive.
- **Rehearsal:** random halves 90.2% (range 88.9% to 90.8% over 20 seeds).
  Temporal halves 91.8%, above the random range, so later calibration
  delays were a little easier to cover than earlier ones. A mild drift of
  about 1.5 points, from one split.
- **Width:** typical upper bound 11.6 minutes (mean 13.2, largest 143)
  against 14 minutes for a bound that ignores the model. 65% of delays get
  a tighter bound than that.

**Against my prediction.** The raw coverage came out at 89.8%, just above my
87.5% to 89.5% range. `q_hat` was positive and small as I expected, but
0.08 minutes, not about 1. The no-model bound was 14, as predicted. My guess
is that validation's 88.5% was partly noise on 2,523 rows; I have not tested
that. `q_hat` uses all 6,563 rows whatever the rehearsal showed, and nothing
was changed because of it.

### Prediction for the test set (written 16 November, before the test measurement)

I expect test coverage of about **90.5%**, within 89% to 92%. If 2026
looked like 2025 it would land in 89.0% to 91.0%. The temporal rehearsal
gave 91.8%, so I lean above 90%: later delays were a bit easier to cover.
But that is one split on calibration data, and the test set sits 9 months
after the training period ends, so drift could go the other way. A result
below 89% would mean 2026 delays ran longer than 2025 predicted.

### Test-set result (Step 7, measured once)

`python src/12_test_once.py --confirm`, once, on 7,448 test delays
(2026-01-01 to 2026-09-30), with `q_hat` = 0.0835 minutes from 6,563
calibration rows.

- **Coverage 92.1%**, 95% CI 91.4% to 92.7%, against a 90% target. The
  expected range if calibration and test are exchangeable was 89.0% to
  91.0%, so the result is above it, and the whole confidence interval is
  above 91.0%.
- **Before conformal:** the model's own q90 covered 91.8%. Conformal added
  0.3 points.
- **No-model bound:** one 14-minute upper bound for every delay covered
  92.9%. The same bound covered at least 90.0% of 2025 calibration delays, so delays
  themselves ran shorter in 2026, not only the model's errors.
- **Width:** upper bound median 11.5 minutes, mean 13.1. 65% of delays got a
  tighter bound than the no-model one.

**Reading it.** Conformal guarantees at least 90%, and that held. But the
result is outside the expected range, so 2025 calibration and 2026 test were
not exchangeable: 2026 delays ran shorter than 2025 predicted, and the
intervals were wider than 2026 needed. This is marginal coverage only; it
says nothing about any line, hour band or delay code. That is Day 4.

**Against my prediction.** I expected about 90.5% (89% to 92%) and leaned
above 90% because the temporal rehearsal gave 91.8%. The direction was
right; the result landed 0.1 points above my range. The rehearsal did flag
this drift.

Nothing was changed after seeing the number. Adaptive conformal inference is
the method built for drift like this; it is a next step for the write-up,
not something applied to this result.

### Coverage figure (Step 8, `13_plot.py`)

Drew `docs/figures/day3_coverage.png` from the saved results, at ten levels
from 50% to 95% fixed before the run. Measured coverage is above the level
asked for at all ten, and none of the 95% intervals crosses zero. The gap is
about 3 to 4 points between 55% and 85%, 2.1 at 90% and 1.4 at 95%. So the
2026 shift is across the whole distribution of scores, not only the upper
tail: the model's bound was too generous at every level. The ten points come
from the same 7,448 delays, so they are one finding, not ten. The figure
adds no new look at the test set. Marginal coverage only.

## Day 4 — 17 November 2026

Goal for the day: test coverage inside each line, hour band and delay-code
tier, each with an interval and a verdict under a rule fixed before the run.
Check, on calibration rows only, whether Mondrian conformal would fix a group
that misses. Nothing measures the test set again: the audit slices the
per-row file Day 3 saved.

### Read

- Angelopoulos and Bates, group-balanced conformal prediction: Mondrian runs
  the Day 3 recipe inside each group, so each group gets its own `q_hat`
  and its own 90% guarantee. A group the global bound undercovers gets a
  bigger `q_hat` and a wider bound; one it overcovers gets a tighter one.
  The cost is that a small group's `q_hat` comes from few rows, so it is
  noisy.
- My Day 1 journal, section 2: the 90% is marginal, a weighted average over
  every delay. A small group can collapse without the overall number
  noticing: if SHP (about 3% of test delays) fell to 80%, the rest would
  only need to average about 92.5% for the overall figure to still read
  92.1%, a shift smaller than its own confidence interval.
- Multiple comparisons and the Bonferroni correction:

### Audit design (Step 1, fixed before any code ran)

**Correction to the plan: Mondrian is not applied to the test set.** The
Day 3 runbook said Day 4 would apply Mondrian conformal if groups
undercovered. If I chose Mondrian after seeing which test groups miss and
then scored it on test, I would be tuning the method to the test set
(rule 2). So I fit and check Mondrian on calibration halves only, and the
write-up reports what it would do. A clean test of it needs delays the
project has never seen, such as months published after September 2026.

I fixed six decisions before any Day 4 code ran:

| Decision | Setting | Why |
|---|---|---|
| Groupings | line, hour band, delay-code tier | Fixed on Day 1. Not station: only 21 of 277 stations reach 100 calibration rows. |
| Code tier | each code's median delay over the 19,804 training rows, cut at 5, 10 and 20 minutes; codes the model pools into `OTHER` form "rare or new code" | One group per code is too thin, and a prefix grouping (MU, SU, PU) would bury `MUPR1` among routine codes. Grouping by typical length tests my Day 1 prediction directly, and because it comes from training rows, coverage cannot leak into it. |
| Thin tiers | a tier under 100 calibration rows merges with its smaller neighbour; "rare or new code" is never merged, and if thin it is shown but not flagged | `MIN_ROWS_PER_GROUP`. Counting calibration rows says nothing about coverage. |
| Intervals | resample whole days of the test period, 2,000 times, seed 0; normal interval around the bootstrap standard error; design effect reported | One bad day produces many long delays together, so rows are not independent. |
| Flag rule | a group undercovers if its Bonferroni-adjusted interval (family-wise 95%) lies entirely below 90%; "possible" if only the ordinary 95% interval does | With about 14 groups tested at 95% each, something flags by chance about half the time. |
| Mondrian | fitted and checked on calibration halves only | The correction above. |

The cutoffs come from Day 1: the overall median is 5 minutes, the 90th
percentile 14, and the long codes run well past 20 (`MUPLB` 25, `MUPR1`
72.5). They are my choice, written down before seeing any group's coverage.
I expect the tier above 20 minutes to be thin: `MUPR1` has 110 delayed rows
over 45 months, so perhaps 20 fall in the nine calibration months. If that
tier merges into the one below, that is the rule working.

### Settings (Step 2)

Added the Day 4 block to the end of `src/config.py`: `AUDIT_GROUPINGS`,
`CODE_TIER_CUTOFFS = (5, 10, 20)`, `AUDIT_BOOTSTRAP_REPS = 2000`,
`AUDIT_SEED = 0`, and the four output paths (`DAY4_GROUPS`,
`DAY4_MONDRIAN`, `DAY4_AUDIT`, `DAY4_AUDIT_TABLE`). The groupings and
cutoffs live in `config.py` with the split dates and the score, so no
script can quietly use different ones. Checked: `config` prints
`('line', 'hour_band', 'code_tier') (5, 10, 20) day4_audit.json`, and
`07_verify.py` still passes all 31 checks, so the edit did not disturb
Day 1.

### Scripts (Step 3)

Copied the seven Day 4 scripts into `src/`: three modules with self-tests
(`groups.py`, `coverage_stats.py`, `mondrian.py`) and four scripts
(`14_define_groups.py`, `15_mondrian.py`, `16_audit.py`,
`17_plot_groups.py`). Before I ran anything, they were checked against my
real schema on a stand-in, with warnings as errors, and I confirmed that
the keys and column types they read are in my Day 3 files. That check found
one fix: `15_mondrian.py` now converts predictions to plain arrays, so the
calibration halves are picked by position even if the model returns pandas
Series. Committed with the design and settings in `f572385`, before any
group was looked at.

### Understanding before running (Step 4)

All seven self-tests passed: the Mondrian hand case at 50% (global `q_hat`
10; A gets 2, B gets 20), Mondrian lifting an undercovered simulated group
to 90%, the Bonferroni z (1.96 for one group, about 2.91 for 14), and the
day bootstrap matching the textbook interval for independent rows but
widening about tenfold when whole days hit or miss together. I deferred the
four hand cases on paper to before the interview because of assignment
deadlines. They are not done yet.

### Groups (Step 5, `14_define_groups.py`)

Built from 19,804 training rows; 6,563 calibration rows were counted, not
scored. Line and hour-band counts match `07_verify.py` exactly. 64 codes
have their own model category; the rest (650 training rows) form "rare or
new code".

**No known code has a training median between 10 and 20 minutes**, so that
tier was empty. Codes sit either at 10 minutes or below, or well above 20.
The tier above 20 minutes had 43 calibration rows, under the 100-row floor,
so the rule merged it into its only neighbour, 5–10 minutes (1,566 rows).

| Code tier | Codes | Training rows | Calibration rows | Medians | Longest codes |
|---|---|---|---|---|---|
| median <= 5 min | 45 | 14,350 | 4,592 | 3 to 5 | `EUBK`, `EUDO`, `EUCD`, `EUNT` |
| median > 5 min | 19 | 4,804 | 1,609 | 6 to 70 | `MUPR1`, `MUPLB`, `SUUT`, `SUEAS` |
| rare or new code | — | 650 | 362 | — | — |

12 groups can be flagged (3 lines, 6 hour bands, 3 code tiers), so the
Bonferroni correction divides by 12.

**What this costs my Day 1 prediction.** The long codes (`MUPR1`, `MUPLB`)
are under 3% of the merged tier's calibration rows, so their coverage is
diluted, just as a small group's is diluted in the overall 92.1%. The audit
cannot test "long-delay codes undercover". Even unmerged, a 43-row tier
would have been description-only, with an interval too wide to show
anything. I am keeping the rule as committed rather than redrawing groups,
and recording this before any group's coverage has been seen: on this data,
the long-code question is out of reach.
This entry was written before Step 6 ran, but it missed the Step 5 commit
(`c41502f` holds only the Step 3 entry). It went in with the Step 7 commit,
after I had seen calibration coverage by group. No test coverage had been
seen.

### Mondrian on calibration (Step 6, `15_mondrian.py`)

Calibration rows only; the global `q_hat` recomputed to 0.0835, matching
Day 3. Temporal split at 2025-08-19, the same as the Day 3 rehearsal. None
of these `q_hat` values is applied to the test set.

**Random halves (exchangeable by construction): Mondrian works.** One
global `q_hat` left three groups below 90%, and Mondrian lifted each to
about 90%:

| Group | Global | Mondrian | Median bound, global → Mondrian |
|---|---|---|---|
| median > 5 min | 87.6% | 89.9% | 20.2 → 23.9 min |
| overnight 00–04 | 88.4% | 90.3% | 13.5 → 15.4 min |
| early 05–06 | 87.8% | 90.1% | 10.9 → 12.8 min |

It also tightened groups the global bound overcovered: rare or new code
92.1% → 90.3% (19.4 → 15.6 min), AM peak 92.4% → 90.2% (10.7 → 9.7 min),
SHP 91.8% → 90.4% (11.0 → 9.0 min). YU and BD sit at about 90% either way:
the line grouping did not need its own `q_hat`.

**Temporal halves (with the 2025 drift): Mondrian made it worse.** Under
the global `q_hat`, every later-half group sat at 90.7% or above except
SHP (87.6%, 129 rows): the drift lifting coverage, as on Day 3. Mondrian
widened the spread across groups in all three groupings (line 4.5 → 11.2
points, hour band 2.5 → 5.3, code tier 1.6 → 6.4). SHP fell to 81.4%,
because its own `q_hat` came from about 145 earlier rows and the later
rows had moved. A per-group `q_hat` fixes imbalance between groups when the
data are exchangeable. It does not fix drift over time, and in a small
group it adds noise. That is the case for adaptive conformal inference, not
Mondrian, as the next step.

**Against earlier predictions (calibration only).** Day 2 said "rare or
new code" would undercover first; on calibration it overcovers. Day 1's
direction for long codes shows up even diluted: the "> 5 min" tier is the
lowest group. Hour band is weaker than code tier, but overnight and early
sit about 2 points low, so not flat.

### Prediction for the audit (Step 7, written before `16_audit.py`)

Test coverage was 92.1% against 90.0% on calibration: 2026 lifted
coverage by about 2 points across the whole distribution. If every group
moves up by about that much, the groups that were low on calibration land
near 90%. To be flagged, a group of about 1,800 test delays needs coverage
below about 88% (adjusted z = 2.87 for 12 groups); smaller groups need
lower still.

- **Groups flagged UNDERCOVERS: none of the 12.** I expect to report a
  negative result.
- **Lowest group: median > 5 min**, about 90% (88.5% to 91.5%). If anything
  is marked "possible", it is this one.
- **Lowest hour bands: early 05–06 and overnight**, about 90% to 91%.
- **Rare or new code: above overall**, about 93%. This reverses my Day 2
  prediction.
- **SHP: hard to call.** Its later calibration half was 87.6% on 129 rows,
  but with 232 test delays its adjusted interval is too wide to flag.
- **Day 1's `MUPR1` prediction cannot be tested** (Step 5).

### Coverage audit (Step 8, `16_audit.py`, run once)

Ran once on 17 November at 16:31, on the 7,448 test delays in Day 3's
per-row file. No model loaded, nothing re-measured. 12 flaggable groups,
adjusted z = 2.87. All three self-tests passed.

**No group undercovers 90% under the rule fixed in advance, and none is
marked "possible".** This is the negative result Day 1 said I would report.

| Grouping | Group | n | Coverage | 95% CI | Adjusted CI | deff |
|---|---|---|---|---|---|---|
| line | YU | 3,977 | 93.1% | 92.2–93.9 | 91.8–94.3 | 1.2 |
| line | BD | 3,167 | 90.7% | 89.5–91.9 | 88.9–92.5 | 1.5 |
| line | SHP | 304 | 93.1% | 90.3–95.9 | 89.0–97.2 | 1.0 |
| hour band | overnight 00–04 | 505 | 89.9% | 87.0–92.8 | 85.6–94.2 | 1.2 |
| hour band | early 05–06 | 646 | 91.3% | 88.7–94.0 | 87.4–95.2 | 1.5 |
| hour band | AM peak 07–09 | 1,194 | 93.3% | 91.8–94.8 | 91.1–95.5 | 1.1 |
| hour band | midday 10–15 | 2,087 | 93.1% | 91.9–94.3 | 91.3–94.9 | 1.2 |
| hour band | PM peak 16–18 | 1,347 | 91.0% | 89.3–92.7 | 88.5–93.5 | 1.3 |
| hour band | evening 19–23 | 1,669 | 91.7% | 90.2–93.1 | 89.5–93.8 | 1.2 |
| code tier | median <= 5 min | 5,373 | 92.8% | 92.0–93.5 | 91.7–93.9 | 1.2 |
| code tier | median > 5 min | 1,528 | 89.6% | 87.9–91.3 | 87.0–92.1 | 1.3 |
| code tier | rare or new code | 547 | 92.0% | 88.5–95.4 | 86.9–97.0 | 2.3 |

Spread (highest minus lowest coverage): line 2.4 points, hour band 3.4,
code tier 3.2.

**Reading it.**

- **Two groups sit just below 90%, but neither interval excludes 90%:** the
  "> 5 min" code tier (89.6%) and overnight (89.9%). They are the same two
  groups one global `q_hat` left low on calibration (87.6% and 88.4% under
  random halves), so the pattern repeats across 2025 and 2026. At this
  sample size it is a pattern, not a finding. On calibration, Mondrian
  lifted both to about 90% for 3.7 and 1.9 extra minutes of median bound.
- **Read against 92.1%, not just 90%:** BD (90.7%) sits 2.4 points below YU
  (93.1%), though the two were level on calibration (90.2% and 90.1%). BD
  gained less from the 2026 shift. I can describe this, not explain it.
- **Power.** The adjusted intervals run about ±1 to ±3 points for the large
  groups and ±4 to ±5 for the small ones. The audit rules out large
  undercoverage (below about 87% in the bigger groups); it cannot rule out
  a shortfall of a point or two.
- **Design effects** are 1.0 to 1.5, except rare or new code at 2.3: its
  misses cluster on particular days. That group is also a bigger share of
  test (547, 7.3%) than of calibration (362, 5.5%), so 2026 brought more
  codes the model never learned.

**Against my prediction.**

| Predicted | Got | |
|---|---|---|
| No group flagged | none flagged, none "possible" | right |
| Lowest group: > 5 min, about 90% (88.5–91.5) | 89.6% | right |
| Lowest hour bands: early and overnight, 90–91% | overnight 89.9% lowest; then PM peak 91.0%, early 91.3% | half right |
| Rare or new code above overall, about 93% | 92.0%, level with overall | wrong, narrowly |
| SHP hard to call | 93.1% | — |
| Day 1: hour band a weak grouping | largest spread of the three (3.4 points) | not supported, though the merge diluted the code tier |
| Day 2: rare or new code undercovers first | 92.0% | not supported |
| Day 1: `MUPR1` undercovers | not testable (Step 5) | — |

**A correction to my earlier entries.** Two numbers in my Read line and my
Step 7 prediction came from Claude's stand-in data, not mine. SHP has 304
test delays (4.1%), not 232 (about 3%). With 4.1%, the rest would need
about 92.6% for the overall figure to read 92.1%, so the conclusion is
unchanged. I left the committed text as written and correct it here.

Nothing was changed after seeing this result. Mondrian stays on
calibration.

### Figure (Step 9, `17_plot_groups.py`)

Drew `docs/figures/day4_group_coverage.png` from the saved audit: one panel
per grouping, a dot for coverage, a thick bar for the 95% interval and a
thin bar for the adjusted interval the flag uses. Every thin bar crosses
the 90% line, matching the table. The "> 5 min" tier and overnight sit
just left of 90%, BD sits below YU, and every group spreads around the
92.1% overall, by up to 3.4 points within a grouping. The figure reads the
saved audit only and adds no new look at the test set.

## Day 5 — 17 and 18 November 2026

Goal for the day: ship what a reviewer reads. The write-up, the README, a
JSON endpoint, and a public repo a stranger can rebuild in two commands.
Nothing is measured again.

### Before starting

`07_verify.py` passed all 31 checks. `12_test_once.py` printed the stored
92.1% and refused to measure again; `16_audit.py` printed the stored audit
(no group undercovers) and refused to run again. The working tree was clean
at `8af606f`, the Day 4 journal commit.

### Log corrections (Step 1)

Changed the Day 3 Step 7 no-model sentence to "at least 90.0%", and removed
the blank Read lines rather than fill them with sources I had not opened.
These went into `ad803ae` with the write-up draft, not a commit of their own.

### Write-up (Step 2)

Claude drafted `docs/writeup.md` from my committed results, log and
journals, and I committed it as drafted (`ad803ae`). Printed from GitHub it
came to 8 pages: site chrome, browser headers and large type. Claude
shortened it and rendered a print PDF instead: 4 pages with both figures and
the audit table, committed in `bdf41ee` (`docs/writeup.pdf`, 351,136 bytes).
No number changed in the cut. Rewriting it in my own words is still to do;
the PDF will be re-rendered from the edited Markdown.

### Endpoint (Step 3)

Added `src/18_serve.py` (9,456 bytes) and `rebuild.py` (1,841 bytes), both
written by Claude and tested on a stand-in with my schema before I ran them.
The endpoint reads `q_hat` from the locked Day 3 result rather than
recomputing it, and refuses to start unless the model was fitted on 19,804
rows with the same score Day 3 measured.

Two examples from the command line; the model knew every station and code:

| Input | Upper bound | Model q90 | Model median |
|---|---|---|---|
| YU, UNION STATION, SUDP, Wed 2026-03-04 08:15 | 9.69 min | 9.609 | 4.3 |
| BD, KENNEDY BD STATION, MUPR1, Sat 2026-03-07 02:30 | 112.24 min | 112.156 | 32.64 |

Same `q_hat` (0.0835) in both, bounds about 12 times apart: the width comes
from the model's 90th percentile, and conformal adds the same small
correction to every delay. The second example sits in the two groups with
the lowest test coverage among lines and hour bands (BD 90.7%, overnight
89.9%), which the endpoint now reports next to the answer.

Server mode: `curl.exe` returned the same JSON as the command line, and the
server logged `200` for each request.

### Rebuild from a fresh clone (Step 5)

**Data window pinned.** `END_DATE` was `None` ("everything available"), so
anyone downloading after October would get rows I never measured, and their
counts would differ from mine. The build keeps `event_time <= END_DATE`, so
the pin is `"2026-09-30 23:59:59"`; my latest row is 23:59:00. Rebuilt: the
new `after end date` stage removed 0 rows, 97,136 rows written, all 31
checks passed. It changes no result; it fixes the data the results were
measured on.

**Requirements pinned.** Listing every import showed two packages my code
uses directly that the old `requirements.txt` never named: `scipy`
(`coverage_stats.py`) and `joblib`. They only worked because scikit-learn
installs them. The new file pins all nine direct dependencies to my
`pip freeze` versions, on Python 3.14.3. `pip install -r requirements.txt`
in my own environment changed nothing.

**Fresh clone.** Cloned from GitHub into a temp folder, new `.venv`,
`pip install -r requirements.txt`, then `python rebuild.py`:

| | My repo | Fresh clone |
|---|---|---|
| Rows in Parquet | 97,136 | 97,136 |
| Acceptance test | 31 passed | 31 passed |
| Final model training rows | 19,804 | 19,804 |
| In-sample share below q90 | 90.6% | 90.6% |
| Example upper bound (YU, UNION, SUDP) | 9.69 min | 9.69 min |

The same model q90 (9.609) means the rebuilt model is the measured model,
so the committed `q_hat` belongs to it. Two transitive packages installed
newer in the clone (narwhals 2.29.0, cloudpickle 3.1.3) without changing
the answer.


### Project page and going public (Steps 6 and 7)

Added `src/19_export_page.py` and `docs/index.html`, both written by Claude.
Every feature the model uses is categorical, so the export asks the final
model for every combination and stores each 90% bound in hundredths of a
minute: `docs/site/bounds.bin`, 1,228,500 bytes, 614,250 answers. It writes
nothing unless the table gives the endpoint's answer for the README example.
The page looks answers up in the browser with no server, and shows both
figures and the locked Day 4 audit. It reads no test row and offers no new
way to slice the test set, so I dropped the free coverage explorer I had
considered: it would have let anyone do the slicing the Day 4 lock exists to
prevent. Committed in `03872ba`.

`git log --all --oneline -- .env .vscode` printed nothing. The repo is
public, and the page is published with GitHub Pages from `/docs`.

### AI assistance

Claude (Anthropic) was used throughout.

**Day 1.** Claude helped scope the project, drafted the pipeline and
exploration scripts, and helped debug the data-loading problems recorded
above. I ran every script myself, read the outputs, and corrected the
analysis where the generated code or its interpretation was wrong: the Q8
spread calculation and the rush-hour hypothesis are both cases where the
output contradicted what I had been told to expect.

**Day 2.** Claude wrote the Day 2 runbook and the four scripts (`data.py`,
`08_baselines.py`, `09_fit_final.py`, `10_leakage.py`), and tested them on
stand-in data matching my Parquet schema before I ran them. The validation
slice, selection rule and feature set were proposed in that runbook; I
checked them against the project's rules before accepting them. I ran every
step on the real data and checked each count against Day 1. Claude drafted
the Day 2 log entries and journal from my pasted outputs, and I reviewed
them.
