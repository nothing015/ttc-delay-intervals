# TTC delay intervals

Japan's railways are famous for running to the minute, which fascinates me.
Instacart takes a different angle on lateness: it uses quantile regression to
estimate an upper bound on delivery time and plans around that bound rather
than the average
([InfoQ, 2017](https://www.infoq.com/news/2017/01/ripert-instacart-ml-optimization)).

After countless delays on the TTC, I wanted to ask the same question of our
own subway: not "how long is a delay on average" but "how long, nine times
out of ten". This is a retrospective analysis rather than a deployable
rider-facing tool.

A rider on a platform does not benefit from "expect 7 minutes." They
benefit from "90% chance service resumes within 14 minutes." This project
produces the second kind of answer, and then checks whether the guarantee
actually holds.

## Quickstart

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then, on either:

```
python src/01_download.py          # fetch raw files from Toronto Open Data
python src/02_build.py             # clean them into one Parquet
python src/03_explore.py           # answer the Day 1 questions, write figures
python src/07_verify.py            # acceptance test: expect "All 31 checks passed"
```

## Layout

```
src/config.py        every shared setting: paths, date range, split dates,
                     modelled lines, hour bands
src/01_download.py   pull the raw files from the CKAN API
src/02_build.py      normalise, validate, stack -> data/processed/delays.parquet
src/03_explore.py    the eight EDA questions, with figures
src/04_splits.py     propose chronological train / calibration / test dates
src/05_groups.py     calibration rows per line, hour band and station
src/06_peek.py       line values, station tail, delay-code key layout
src/07_verify.py     Day 1 acceptance test; exits 1 on any failure
docs/log.md          running notes, one entry per day
docs/day1_journal.md Day 1 write-up: decisions, corrections, concepts
docs/figures/        plots written by 03_explore.py
```

Scripts are numbered in execution order. Each one is safe to re-run.

## Data

[TTC Subway Delay Data](https://open.toronto.ca/dataset/ttc-subway-delay-data/),
City of Toronto Open Data Portal, under the Open Government Licence –
Toronto. One row per logged delay-causing incident.

The prediction target is `min_delay` — minutes of delay to service —
**conditional on an incident having been logged**. This does not predict
whether a delay will occur, only how long one lasts once it has started.

Modelling population: delays of at least one minute on Lines 1, 2 and 4
(YU, BD, SHP), January 2023 to September 2026 — 33,815 rows. The Line 3
SRT (closed July 2023) and a handful of bus and streetcar labels are
excluded; see `docs/log.md`.

Splits are by date, never random: train before 2025-04-01, calibration
2025-04-01 to 2025-12-31, test from 2026-01-01.

## Quality checks

Built into `02_build.py`, each reporting how many rows it affected:

1. **Schema check** — required columns present in *each* file after alias
   normalisation; fatal if not, and names the file
2. **Range check** — `min_delay` non-negative and at most 720 minutes
3. **Duplicate check** — exact repeats on time, station, code and delay
4. **Row accounting** — rows in and rows out printed at every stage

`07_verify.py` then asserts what later days depend on: column types, the
derived `hour` column, split integrity (disjoint, complete, chronological),
group sizes, and the delay-code key.

## Status

Day 1 complete: data pipeline, exploration, splits and acceptance test.
Modelling starts Day 2.
