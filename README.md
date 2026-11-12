# TTC delay intervals

Predicting how long a Toronto subway delay will last — as a calibrated
interval with a coverage guarantee, not a single number.

A rider on a platform does not benefit from "expect 7 minutes." They
benefit from "90% chance service resumes within 14 minutes." This project
produces the second kind of answer, and then checks whether the guarantee
actually holds.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/01_download.py          # fetch raw files from Toronto Open Data
python src/02_build.py             # clean them into one Parquet
python src/03_explore.py           # answer the Day 1 questions
```

## Layout

```
src/config.py        every shared setting: paths, date range, split dates
src/01_download.py   pull the raw XLSX files from the CKAN API
src/02_build.py      normalise, validate, stack -> data/processed/delays.parquet
src/03_explore.py    the eight EDA questions, with figures
docs/log.md          running notes, one entry per day
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

## Quality checks

Built into `02_build.py`, each reporting how many rows it affected:

1. **Schema check** — required columns present after alias normalisation; fatal if not
2. **Range check** — `min_delay` non-negative and below a plausible ceiling
3. **Duplicate check** — exact repeats on time, station, code and delay
4. **Row accounting** — rows in and rows out printed at every stage

## Status

Day 1 complete: data pipeline and exploration. Modelling starts Day 2.
