# Project log

Japan has public transport scheduling which is always on time - which is fascinating.
On the other hand Instacart's business model works how it engineers quantile regression into its system.


I wanted to run an analysis on our very own TTC subway lines based on the aforementioned moivation, after facing countless delays..
This is a retrospective analysis rather than a deployable rider-facing
tool

## Day 1

Goal for the day: raw files in, one clean Parquet out, the prediction target
written down. Nothing modelled.

### Read

- Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction*
  (introduction and split-conformal section):
- TTC dataset readme and delay-code key:
- Open Government Licence – Toronto:
- Leakage in time series (optional):

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
all lines).

Any code-based feature therefore needs an explicit "unknown code" category
rather than an assumption that the key is complete. This is quality-check
material for Part 4.

### Target paragraph

The population is subway delay incidents logged by the TTC on Line 1 (YU),
Line 2 (BD) and Line 4 (SHP) between 1 January 2023 and 30 September 2026,
restricted to incidents where the recorded delay was at least one minute —
34,163 of the 97,136 logged records. The target is `min_delay`, the minutes
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

## Day 2
Correction 1: baselines are not scored on the test set. The plan says "scored on a temporal test set". That breaks the project's own rule: the test set is measured once, at the end. The calibration set cannot be used either, because Day 3 needs it untouched for the conformal quantile. So choosing between baselines needs a fourth slice, cut from the end of the training period:

[ fit ................ | valid ] [ calibration ] [ test ]
2023-01-01        2025-01-01  2025-04-01      2026-01-01

Correction 2: a plain station-by-hour median does not work. Only a few dozen stations have enough rows. The baseline uses a fallback instead: station and hour band if that cell has at least 30 training rows, otherwise line and hour band, otherwise line, otherwise the overall median.

### Selection Rule (Step 5)
I will choose the baseline with the lowest pinball loss at the 90th percentile on the validation slice.
- Writing it first means the choice cannot drift toward whichever number looks best afterwards. Why the 90th rather than the 50th: the deliverable is "90% chance it clears within X", which is an upper bound.


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

### AI assistance

Claude (Anthropic) was used to scope the project, draft the pipeline and
exploration scripts, and help debug the data-loading problems recorded above.
I ran every script myself, read the outputs, and corrected the analysis where
the generated code or its interpretation was wrong — the Q8 spread
calculation and the rush-hour hypothesis are both cases where the output
contradicted what I had been told to expect. Reading, decisions and this log
are my own (draft skeleton by CLaude).
