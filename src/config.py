"""
Every setting the project shares lives here.

Why a file and not numbers typed into each script: the split dates in
particular get read by Day 2, Day 3 and Day 4 code. If they are typed in
three places they will eventually disagree, and a disagreement between the
calibration split and the test split silently destroys the coverage
guarantee. One definition, imported everywhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------
# Path(__file__) is this file. .resolve() makes it absolute. .parent is src/,
# .parent.parent is the project root. Doing it this way means the scripts work
# no matter which directory you run them from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

DELAYS_PARQUET = PROCESSED_DIR / "delays.parquet"

# ---------------------------------------------------------------------------
# The dataset on Toronto's open data portal
# ---------------------------------------------------------------------------
CKAN_BASE = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action"
PACKAGE_ID = "ttc-subway-delay-data"

# ---------------------------------------------------------------------------
# Which rows we model
# ---------------------------------------------------------------------------
# Decided on Day 1. See docs/log.md for the reasoning.
#
# START_DATE: the TTC system changed a lot over twelve years (pandemic
# ridership collapse, the Line 3 Scarborough RT closure). Training on a long
# span of different regimes makes the model worse, not better. 2023 onward is
# a recent, reasonably stable period.
START_DATE = "2023-01-01"
END_DATE = None  # None means "everything available"

# Minimum delay in minutes for a row to count as an actual delay. Most logged
# incidents caused no measurable delay; those are not what we are predicting.
MIN_DELAY_THRESHOLD = 1

# ---------------------------------------------------------------------------
# The three-way split, by date
# ---------------------------------------------------------------------------
# Chronological, never random. Train is the earliest block, test is the
# latest. Rows before TRAIN_END are training; rows from TRAIN_END up to
# CALIB_END are calibration; everything after CALIB_END is test.
#
# These are placeholders until Day 1 Block D, where you set them from the
# actual date range in your data. Aim for roughly 60 / 20 / 20.
TRAIN_END = "2025-04-01"
CALIB_END = "2025-11-01"

# The coverage level we are targeting. 0.90 means "90% of the time the true
# delay should fall inside the interval".
TARGET_COVERAGE = 0.90

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# The TTC's Excel files do not agree with each other across years. Column
# names drift: spacing, underscores, capitalisation, and "Report Date" versus
# "Date". We map every variant we have seen onto one canonical name.
#
# Keys here are ALREADY normalised (lowercased, stripped, spaces and hyphens
# turned into underscores) by _normalise_name() in 02_build.py. So you only
# need an entry when the *word* differs, not the punctuation.
COLUMN_ALIASES = {
    "report_date": "date",
    "date": "date",
    "time": "time",
    "day": "day",
    "station": "station",
    "code": "code",
    "min_delay": "min_delay",
    "mindelay": "min_delay",
    "delay": "min_delay",
    "min_gap": "min_gap",
    "mingap": "min_gap",
    "gap": "min_gap",
    "bound": "bound",
    "line": "line",
    "vehicle": "vehicle",
}

# After renaming, these columns must all be present or the build stops.
# This is quality check #1 of the four the project needs.
REQUIRED_COLUMNS = [
    "date",
    "time",
    "station",
    "code",
    "min_delay",
    "min_gap",
    "line",
]

# Sanity bounds. A delay of 1200 minutes (20 hours) is not a delay, it is a
# data-entry error. Quality check #2.
MAX_PLAUSIBLE_DELAY_MIN = 720  # 12 hours
