# Day 1 journal: data in, problem pinned down

**Project:** prediction intervals for TTC subway delay length, with a coverage guarantee
**Dates:** 13–14 November 2026
**Status:** done. `python src/07_verify.py` passes all checks.

This is my record of Day 1: what I built, what broke, what I decided and why.
I also use it as revision, so each section is written so that I can answer
questions about it without opening the code.

---

## 1. Day 1 in one minute

| | |
|---|---|
| Goal | Raw TTC files in, one clean Parquet out, the prediction target written down. Nothing modelled. |
| Output | `data/processed/delays.parquet`, 97,136 rows, 2023-01-01 to 2026-09-30 |
| Modelling population | 33,815 delays of at least 1 minute on Lines 1, 2 and 4 |
| Headline number | Median delay 5 minutes, 90th percentile 14 minutes |
| Splits | Train before 2025-04-01, calibration to 2025-12-31, test from 2026-01-01 |
| Biggest surprise | My rush-hour hypothesis was wrong |
| Biggest save | A file named `.xlsx` that was really CSV was hiding 47,915 rows |

The one sentence that justifies the project: **a rider told "about 5 minutes"
waits 14 or more one time in ten.** A single number hides that. An interval
does not.

---

## 2. Concepts I need to be able to explain

### Point forecast vs prediction interval

A point forecast says "7 minutes". A prediction interval says "between 2 and
14 minutes, and I am right 90% of the time". The second is what a person on a
platform can act on. My deliverable is the interval and the evidence that the
90% is real.

### Split conformal prediction, in five steps

1. Fit any model on the **training** set.
2. On the **calibration** set (never trained on), compute how wrong the model
   was for each row: `score = |actual − predicted|`.
3. Take roughly the 90th percentile of those scores. Call it `q`. The exact
   level is `ceil((n + 1) × 0.90) / n`. With my 6,565 calibration rows that is
   5,910 / 6,565 = 0.9002, so almost exactly 90%.
4. For a new case, the interval is `prediction ± q`.
5. The theorem: this interval contains the truth at least 90% of the time,
   **whatever the model is**, as long as the calibration and test data are
   exchangeable.

A bad model still gives valid intervals. They are just uselessly wide. The
model decides the width; conformal decides the honesty.

### Why three slices instead of two

The guarantee comes from the calibration errors looking like the errors the
model will make on new data. A model's errors on its own training rows are
too small, because it has already seen those answers. If I calibrated on
training data, `q` would be too small, the intervals too narrow, and real
coverage would fall below 90% **with no error message**. So calibration rows
are never trained on. That is rule number one of the project.

### Why I split by date and never at random

The data is ordered in time. With a random split the model could learn from
next Tuesday to predict last Tuesday. That is leakage, and it makes results
look better than they are. Splitting by date copies real use: train on the
past, predict the future.

### Why the test set is touched once

If I look at test coverage, change something, and look again, I am tuning to
the test set and the number stops meaning what it claims. All tuning happens
against calibration. The test set is measured once, at the end.

### Marginal vs conditional coverage

- **Marginal:** 90% averaged over everything. This is what conformal
  guarantees.
- **Conditional:** 90% inside every subgroup (each line, each hour band, each
  delay code). Conformal does **not** guarantee this.

Coverage can be 90% overall and 70% for one type of delay, balanced by 95%
elsewhere. Day 4 exists to look for that.

### Mondrian (group-conditional) conformal

The fix when a subgroup undercovers: compute a separate `q` inside each group
instead of one global `q`. The cost is that every group needs enough
calibration rows for its own quantile.

### Why about 100 rows per group

The error of an estimated quantile shrinks like `1 / √n`. With 14 rows a 90th
percentile is noise. I use 100 as a floor, not a target
(`MIN_ROWS_PER_GROUP` in `config.py`).

### Exchangeability

It means the order of the data carries no information, so calibration rows
and test rows are interchangeable. Time series break this, because 2025 and
2026 can differ. I do not solve this. I name it, test for drift later, and
cite adaptive conformal inference as the method that addresses it.

### Small statistics I use

- **Median:** the middle value. Robust to huge outliers.
- **p90:** 90% of values are at or below it.
- **IQR:** 75th percentile minus 25th. My measure of spread.
- **Skew:** 21.5 on raw delays (a very long right tail), 1.82 after `log1p`.
  That is why I will model `log1p(min_delay)` and convert back to minutes.

### Tools

- **Virtual environment (`.venv`):** a private set of packages for this
  project only. Must be activated in every new terminal.
- **Parquet:** a compressed column file. Loads far faster than Excel and
  remembers types, so dates stay dates.
- **`config.py`:** every shared setting lives there once. If split dates were
  typed in three scripts they would eventually disagree, and that would break
  the guarantee silently.

---

## 3. Setup on Windows, and what tripped me up

```powershell
cd C:\Users\nnuai\Downloads\ttc-delay-intervals
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

| Problem | Cause | Fix |
|---|---|---|
| `python3` not found | My install only has `python` | Use `python` everywhere |
| `source .venv/bin/activate` fails | That is the macOS/Linux form | `.\.venv\Scripts\Activate.ps1` |
| "running scripts is disabled" | PowerShell blocks scripts by default | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (this window only, nothing permanent) |
| Multi-line `python -c "..."` breaks | PowerShell mangles the quoting | Keep it on one line, or put it in a `.py` file |
| `ModuleNotFoundError` on an installed package | Forgot to activate `.venv` | Activate, check the prompt shows `(.venv)` |

---

## 4. The pipeline

| Script | What it does |
|---|---|
| `01_download.py` | Asks Toronto's open data API (CKAN) for the file list and downloads each into `data/raw/`. Safe to run twice. |
| `02_build.py` | Reads the raw files, fixes their differences, runs the quality checks, writes the Parquet. Prints a row count at every stage. |
| `03_explore.py` | Answers eight fixed questions and saves figures to `docs/figures/`. |
| `04_splits.py` | Proposes split dates from the 60% and 80% row quantiles, rounded to month starts. |
| `05_groups.py` | Counts calibration rows per line, hour, hour band and station. |
| `06_peek.py` | Lists the `line` values, the station tail and the code-key file layout. |
| `07_verify.py` | Acceptance test. Asserts everything Day 2 depends on. Exits 1 on failure. |

The build's row accounting on the real data:

```
read 2023 file                  22,949
read 2024 file                  26,467
read 2025-onward CSV            47,915
TOTAL read                      97,331
after range check on min_delay  97,328   -3
after dropping duplicates       97,136   -192
```

Every filter reports what it removed. Nothing is dropped silently.

The four quality checks in the build: schema (required columns present, per
file), plausible range (0 to 720 minutes), duplicates (same time, station,
code and delay), and the row accounting itself.

---

## 5. What went wrong, and what each one taught me

This is the part I most want to remember.

### Data loading

**1. A file named `.xlsx` was really CSV text.**
`TTC_Subway_Delay_Data_since_2025.xlsx` failed with "Excel file format cannot
be determined". The first build finished anyway with 49,347 rows and a
plausible-looking total, but everything from 2025 on was missing. The build
now checks what a file contains (a real Excel file starts with the bytes
`PK`) instead of trusting its name. `Code_Descriptions.xlsx` has the same
problem.
*Lesson: a pipeline that finishes is not a pipeline that is right. Check the
date range of the output, not just that there is output.*

**2. Mixed date formats became blanks when files were stacked.**
pandas guesses one date format from the first rows and turns every other
format into `NaT` (missing). The 2023 and 2024 files hold Excel dates; the
2025 file holds text like `2025-01-01T00:00:00`. Parsing after stacking wiped
out thousands of timestamps without any error. Now each file builds its own
`event_time` before stacking.
*Lesson: parse per file, then combine.*

**3. The schema check was in the wrong place.**
It ran on the stacked table. If one file lacked `min_delay`, the other files
supplied the column and the check passed, while that file's rows were lost.
It now runs per file and names the file. I proved it by renaming `MIN_DELAY`
to `MIN_DELAYX` in the 2025 CSV: the build stopped with
`SCHEMA CHECK FAILED in TTC_Subway_Delay_Data_since_2025.csv`.
*Lesson: an assertion nobody has seen fail is not a safety net.*

**4. Some older workbooks have one sheet per month.**
The 2018 to 2021 files would have been read as January only. They are
outside my date window, so nothing was lost, and the build now warns if it
ever reads a multi-sheet workbook.

**5. pandas 3 deprecations.** `ExcelFile.parse` became `pd.read_excel`, and
`dayofweek` became `day_of_week`.

**6. pandas 3 keeps missing values missing.** `.astype(str)` used to turn a
blank into the text `"nan"`. Now it stays missing. Two consequences:
sorting a column that mixes text and a missing value raises `TypeError`, and
`groupby` silently drops rows whose key is missing. I now check null counts
before grouping. In my population there are none (2 rows had no line and are
excluded).

### Analysis

**7. My rush-hour hypothesis was wrong.**
The plan assumed delays at rush hour are longer and more variable, so one
global interval width would fail there. The data says no:

| | rows | median | IQR | p90 |
|---|---|---|---|---|
| Rush hour (7–9, 16–18) | 11,462 | 4 | 4 | 13 |
| Off-peak | 22,701 | 5 | 4 | 15 |

Same spread, and the 90th percentile is slightly lower at rush hour. Hour by
hour, the morning peak is the tightest part of the day (IQR 3) and late
evening to after midnight is the widest (IQR 5 to 7).
*Lesson: write the hypothesis down before looking, then let the data win.*

**8. The first version of that check was itself wrong, twice.**
- It averaged the 24 hourly IQRs, so hour 3 (14 rows) had the same vote as
  hour 8 (2,224 rows). That gave a misleading ratio of 0.75. Pooling the raw
  rows gives 1.00.
- It only raised a flag when rush hour was wider, so a result in the other
  direction printed nothing. It now reports wider, narrower or similar.

*Lesson: never average statistics from groups of very different sizes, and a
check should speak in every direction.*

**9. Three guesses that the data corrected.**
- The one-row line in the calibration slice was assumed to be SRT. It was
  `62 MORTIMER`, a bus route. SRT ended in July 2023.
- The long tail of station names was assumed to be noise. It is mostly real
  stations with notes attached, such as `VMC STATION PLATFORM 1` and
  `KENNEDY BD STATION [PL`.
- "54 codes missing from the key" mostly measured the SRT. After dropping
  SRT rows the figure is 5 codes (see section 6).

*Lesson: label a guess as a guess until a command has confirmed it.*

**10. The acceptance test needed three attempts.**
- It demanded `date` and `time` columns, but the build folds those into
  `event_time` and drops them.
- It crashed on a text `event_time` instead of failing cleanly.
- It crashed sorting line names that included a missing value.

Each time, the test data used to try the script had been built from what the
script expected, not from what the real pipeline writes, so the test agreed
with the bug.
*Lesson: test data must mirror the real file, including its awkward parts.*

---

## 6. What the data looks like

**Size.** 97,136 logged incidents over 45 months. Only 35.2% (34,163) caused
a delay of at least one minute. Monthly counts run from 1,736 to 2,715 with
no thin months.

**Delays.** Median 5, p90 14, p99 46, maximum 716 minutes. The 720-minute
ceiling removed only 3 rows, so it is a guard against typing errors, not a
tuned threshold.

**Delay codes.** 197 codes in the data. Ranking by count and by total minutes
gives different lists, and the tail is where the long delays live:

| Code | Incidents | Median delay |
|---|---|---|
| `SUDP` (most common) | 4,456 | 5 min |
| `MUPLB` | 196 | 25 min |
| `MUPR1` | 110 | 72.5 min |

**Code key is incomplete.** `Code_Descriptions.csv` lists 140 codes. Across
all delayed rows, 54 codes are missing from it (778 rows, 2.3%), but most of
those are SRT equipment codes kept in a different file. In my modelling
population it is **5 of 129 codes, 1.3% of rows**. The largest is `PUTO`.

| Code | Rows in population |
|---|---|
| `PUTO` | 281 |
| `SUL` | 121 |
| `MUFL` | 17 |
| `MUBIO` | 17 |
| `TUAV` | 1 |

**Lines.** The `line` column has 13 values, not 4:

| Value | Delayed rows | What it is |
|---|---|---|
| YU | 19,318 | Line 1 |
| BD | 13,268 | Line 2 |
| SHP | 1,225 | Line 4 |
| SRT | 339 | Line 3, last row 2023-07-24 |
| YUS, BD LINE 2 | 3, 1 | Other spellings of YU and BD |
| 7 route names | 1 each | Buses and streetcars (`504 KING`, `506 CARLTON`, ...) |

**Stations.** 610 different station strings among delayed rows for about 70
real stations. 76 of them cover 94.5% of rows. Cleaning case and spaces
merged nothing, so the variants are different text, not formatting.

---

## 7. Decisions, each with its reason

| Decision | Reason |
|---|---|
| Data from 2023-01-01 | The system changed a lot over twelve years (pandemic, SRT closure). Mixing regimes hurts the model and breaks exchangeability. Monthly counts since 2023 are stable. |
| Target is `min_delay` ≥ 1 | 64.8% of rows are zero-delay log entries. Those are not what a rider experiences. |
| Model only YU, BD, SHP | SRT is closed and sits entirely in the training period. Bus and streetcar labels are the wrong dataset. 348 rows dropped (1.0%). |
| Remap YUS → YU, BD LINE 2 → BD | Same lines, different spelling. Remapped, not dropped. |
| Filter lines at modelling time, not in the build | The Parquet stays a faithful copy of the published data. Exclusions are a modelling choice I state openly. |
| TRAIN_END 2025-04-01, CALIB_END 2026-01-01 | Nearest month starts to the 60% and 80% row quantiles. |
| Six hour bands, not 24 hours | Hours 2, 3 and 4 hold 141, 14 and 21 delayed rows in the whole dataset. Every band has at least 442 calibration rows. |
| Station is not a conformal group | Only 21 of 277 stations reach 100 calibration rows. Groups will be line, hour band and code category. |
| Code features need an "unknown" category | The official key is incomplete. |

Split sizes over all 34,163 delayed rows (verified disjoint, complete and in
time order):

| Slice | Rows | Share | Covers |
|---|---|---|---|
| Train | 20,146 | 59.0% | to 2025-03-31 |
| Calibration | 6,565 | 19.2% | 2025-04-01 to 2025-12-31 |
| Test | 7,452 | 21.8% | from 2026-01-01 |

Hour bands in the calibration slice: midday 10–15 (1,756), evening 19–23
(1,474), PM peak 16–18 (1,180), AM peak 07–09 (974), early 05–06 (737),
overnight 00–04 (442).

---

## 8. The target, in my own words

I am predicting how long a subway delay lasts, in minutes, **given that the
TTC has logged one**. The population is delays of at least one minute on
Lines 1, 2 and 4 between January 2023 and September 2026. The output is a 90%
prediction interval. At prediction time I assume the station, line, time, day
and the TTC's reason code are known.

What I am **not** claiming:

- I do not predict whether a delay will happen.
- The reason code is assigned by TTC staff, so a rider would not have it at
  the moment the train stops. This makes the work retrospective, not a
  deployable rider tool.
- The 90% is marginal. I do not claim it holds inside every subgroup. Day 4
  tests exactly that.

---

## 9. My prediction for Day 4 (written before any modelling)

- Marginal coverage on the test set will be within a point or two of 90%.
- Undercoverage, if it appears, will be in rare long-delay code groups such
  as `MUPR1`, because one global width cannot fit both a 5-minute delay and a
  70-minute shutdown.
- Hour of day will be a weak grouping.
- If nothing undercovers once I put confidence intervals on the coverage
  numbers, I will report that as a negative result and not keep slicing until
  something breaks.

---

## 10. Questions I should be able to answer quickly

**Why does conformal work without the model being correct?**
It never assumes the model is right. It measures how wrong the model is on
held-out data and widens the interval by that amount. A worse model gives a
wider interval, not an invalid one.

**What is the one assumption?**
Exchangeability between calibration and test data. Mine is a time series, so
it holds only approximately, and I say so.

**Why a calibration set separate from training?**
Training errors are too small. Using them would make intervals too narrow
and coverage would quietly fall below 90%.

**Why not a random split?**
Leakage from the future into the past. It inflates results.

**Does 90% coverage hold for every station or hour?**
No. It is an average over everything. Subgroups can be well below it.

**Why 2023 onward when twelve years exist?**
Different operating regimes. More data from a different system is not better
data.

**What surprised you?**
Rush hour is not more variable than off-peak. I expected the opposite.

**What was the worst bug?**
A mislabelled file silently removed every row from 2025 onward and the build
still looked successful.

**How do you know your splits are sound?**
`07_verify.py` asserts the three slices are disjoint, cover every row, and
are strictly in time order. Train ends 2025-03-31 23:43 and test starts
2026-01-01 00:12.

**Why is your target conditional on a logged delay?**
Because that is what the data records. Predicting whether a delay occurs
would need data on trips with no incident, which this dataset does not have.

---

## 11. Commands I run to check everything

```powershell
python src/02_build.py     # expect 97,136 rows
python src/03_explore.py   # the eight answers and the figures
python src/05_groups.py    # calibration rows per group
python src/07_verify.py    # expect "All 31 checks passed"
```

Deleting the Parquet and running `02_build.py` gives the same 97,136 rows, so
the build is reproducible from the raw files.

---

## 12. Open items

- [ ] Paste the five undocumented codes into section 6
- [ ] Look at `docs/figures/q3_incidents_per_month.png` and confirm no gaps
- [ ] Test `requirements.txt` in a fresh environment (Day 5)
- [ ] Day 2 starts with: load the Parquet, apply `MODEL_LINES` and
      `LINE_ALIASES`, build the station-hour median baseline

---

## AI acknowledgement

*(to be written by me)*
