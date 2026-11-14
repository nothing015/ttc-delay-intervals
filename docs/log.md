# Project log

## Day 1 — 13–14 November 2026

Goal for the day: raw files in, one clean Parquet out, the prediction target
written down. Nothing modelled.

### Read

> FILL IN — one line each: what you read, and the single thing you took from it.
> Leave out anything you did not actually read. An honest short list beats a
> long one you cannot discuss in an interview.

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

> FILL IN — paste the five codes and row counts that `07_verify.py` prints
> under section 4. I have not named them here because I could not verify
> which of the codes in the first run survive the line filter.

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

### AI assistance

> FILL IN / EDIT — be specific and accurate; vagueness here reads worse than
> the honest version.

Claude (Anthropic) was used to scope the project, draft the pipeline and
exploration scripts, and help debug the data-loading problems recorded above.
I ran every script myself, read the outputs, and corrected the analysis where
the generated code or its interpretation was wrong — the Q8 spread
calculation and the rush-hour hypothesis are both cases where the output
contradicted what I had been told to expect. Reading, decisions and this log
are my own.
