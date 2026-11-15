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
# Set on Day 1 from the real date range (2023-01-01 to 2026-09-30) by taking
# the nearest month start to the 60% and 80% row quantiles. Slice counts are
# in docs/log.md. Changing these invalidates every result from Day 3 onward,
# so they are not to be nudged after the test set has been measured.
TRAIN_END = "2025-04-01"
CALIB_END = "2026-01-01"

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


# ---------------------------------------------------------------------------
# Which lines we model, and the groups for the Day 4 audit
# ---------------------------------------------------------------------------
# Decided on Day 1 from 06_peek.py. The `line` column holds more than the
# four subway lines:
#   SRT             Line 3 Scarborough RT. 339 delayed rows, last one
#                   24 July 2023, so it sits entirely inside the training
#                   period and contributes nothing to calibration or test.
#                   Its equipment codes (ER*, SR*) are also absent from the
#                   main code key. Dropped.
#   YUS, BD LINE 2  Alternate spellings of YU and BD. Remapped, not dropped.
#   62 MORTIMER, 506 CARLTON, 504 KING, 20 CLIFFSIDE, 109 RANEE,
#   76 ROYAL YORK SOUTH, 77 SWANSEA
#                   Bus and streetcar routes, one row each. Wrong mode of
#                   transport for this dataset. Dropped.
MODEL_LINES = ("YU", "BD", "SHP")

LINE_ALIASES = {
    "YUS": "YU",
    "BD LINE 2": "BD",
}

# Hour bands for group-conditional (Mondrian) conformal on Day 4. Raw hours
# do not work: hours 2, 3 and 4 hold 141, 14 and 21 delayed rows in the
# entire dataset, far too few to estimate a 90% quantile. Verified with
# 05_groups.py, where every band below clears 400 calibration rows.
HOUR_BANDS = [
    ("overnight 00-04", range(0, 5)),
    ("early 05-06", range(5, 7)),
    ("AM peak 07-09", range(7, 10)),
    ("midday 10-15", range(10, 16)),
    ("PM peak 16-18", range(16, 19)),
    ("evening 19-23", range(19, 24)),
]

# How many calibration rows a group needs before its own 90% quantile means
# anything. The empirical quantile's error scales like 1/sqrt(n), so this is
# a floor, not a target.
MIN_ROWS_PER_GROUP = 100

# The delay-code key that 03_explore.py joins against. Code_Descriptions.csv
# has clean CODE and DESCRIPTION columns. Note that Code_Descriptions.xlsx is
# CSV text with an .xlsx name, and ttc-subway-delay-codes.xlsx puts the
# subway and SRT code lists side by side in one sheet, so neither is used.
CODE_KEY_FILE = "Code_Descriptions.csv"

# ---------------------------------------------------------------------------
# Day 2: choosing a baseline without touching calibration or test
# ---------------------------------------------------------------------------
# Comparing baselines needs held-out data, but the calibration slice is
# reserved for Day 3's conformal quantile and the test slice is measured
# once at the end. So the comparison uses a VALIDATION slice cut from the
# end of the training period: rows from VALID_START up to TRAIN_END.
#
#   [ fit ............ | valid ] [ calibration ] [ test ]
#   2023-01-01     2025-01-01  2025-04-01    2026-01-01
#
# Once a baseline is chosen it is refitted on fit + valid (all of training)
# before Day 3. Calibration and test are never used to choose anything.
VALID_START = "2025-01-01"

# The quantiles every baseline predicts. 0.5 is the typical delay; 0.9 is
# the upper bound a rider cares about ("90% chance it clears within X").
QUANTILES = (0.5, 0.9)

# A station or delay code needs this many FIT rows to get its own category.
# Rarer ones are pooled into OTHER. Learned from the fit slice only.
MIN_CATEGORY_ROWS = 30

# Fitted models are written here. Ignored by git: they are rebuilt by script.
MODELS_DIR = PROJECT_ROOT / "models"