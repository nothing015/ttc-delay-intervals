"""
Turn the raw TTC Excel files into one clean Parquet file.

Run it with:
    python src/02_build.py

Reads everything in data/raw/, normalises it, stacks it, validates it,
writes data/processed/delays.parquet, and prints a row count at every
stage so you can see exactly where rows went.

WHAT THIS SCRIPT IS FOR
-----------------------
The raw files are hand-maintained Excel spanning years, so they disagree
with each other. This script is the one place that knows about those
disagreements. Everything downstream reads the Parquet and can assume it
is clean.

Design rule followed throughout: never drop a row silently. Every row that
disappears is counted and reported. That row accounting is one of the four
quality checks the project needs, and it is the thing that makes the
pipeline trustworthy.
"""

import sys

import pandas as pd

from config import (
    COLUMN_ALIASES,
    DELAYS_PARQUET,
    END_DATE,
    MAX_PLAUSIBLE_DELAY_MIN,
    PROCESSED_DIR,
    RAW_DIR,
    REQUIRED_COLUMNS,
    START_DATE,
)


# A running tally of rows at each stage. Printed at the end.
row_log = []


def log_stage(stage, n_rows, note=""):
    """Record a row count so we can print the full accounting at the end."""
    row_log.append({"stage": stage, "rows": n_rows, "note": note})
    print(f"  {stage:<42} {n_rows:>9,}  {note}")


def normalise_name(name):
    """Turn a raw column header into a predictable lowercase token.

    '  Min Delay ' -> 'min_delay'
    'Report Date'  -> 'report_date'

    This handles punctuation and casing. The COLUMN_ALIASES dict in
    config.py then handles the cases where the actual *word* differs.
    """
    text = str(name).strip().lower()
    for char in (" ", "-", "/", "."):
        text = text.replace(char, "_")
    # Collapse runs of underscores: 'min__delay' -> 'min_delay'
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def is_delay_file(path):
    """True if this looks like a delay data file rather than the readme.

    The readme and the code lookup live in the same folder; we load those
    separately.
    """
    stem = path.stem.lower()
    if "readme" in stem or "code" in stem:
        return False
    return path.suffix.lower() in {".xlsx", ".xls"}


def load_one_file(path):
    """Read one Excel file and rename its columns to the canonical set.

    Returns a DataFrame, or None if the file could not be used.
    """
    try:
        # sheet_name=0 reads the first worksheet. Some TTC files have one
        # sheet per month; if you hit one, sheet_name=None returns a dict of
        # every sheet and you concatenate those.
        frame = pd.read_excel(path, sheet_name=0)
    except Exception as error:
        print(f"    !! could not read {path.name}: {error}")
        return None

    # Step 1: normalise punctuation and casing on every header.
    frame.columns = [normalise_name(c) for c in frame.columns]

    # Step 2: map known variants onto canonical names.
    rename_map = {c: COLUMN_ALIASES[c] for c in frame.columns if c in COLUMN_ALIASES}
    frame = frame.rename(columns=rename_map)

    # Step 3: report anything we did not recognise. This is deliberate -
    # a new column appearing is information, not noise. We keep going, but
    # you get told.
    unknown = [c for c in frame.columns if c not in COLUMN_ALIASES.values()]
    if unknown:
        print(f"    note: {path.name} has unrecognised columns: {unknown}")

    frame["source_file"] = path.name
    return frame


def check_schema(frame):
    """Stop the build if a required column is missing.

    QUALITY CHECK 1 of 4. Deliberately fatal: a missing column means the
    file format changed, and quietly carrying on would produce a Parquet
    that looks fine and is wrong.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        print(f"\nSCHEMA CHECK FAILED. Missing columns: {missing}")
        print(f"Columns actually present: {sorted(frame.columns)}")
        print("\nAdd the new spelling to COLUMN_ALIASES in config.py.")
        sys.exit(1)
    print("  schema check: all required columns present")


def build_event_time(frame):
    """Combine the separate date and time columns into one timestamp.

    This is the fiddliest part of the whole script. The TTC stores the date
    in one column and the time in another, and the time arrives in at least
    three different shapes depending on the file:

      - a string like '08:42'
      - a datetime where only the time part is meaningful
      - an Excel serial fraction (0.3625 == 08:42)

    Strategy: convert both to strings, glue them together, let pandas parse
    the result. Anything that fails becomes NaT (not-a-time) rather than
    crashing, and we count the failures.
    """
    date_text = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    time_raw = frame["time"]

    # If the time column came through as datetime-like, take its time part.
    if pd.api.types.is_datetime64_any_dtype(time_raw):
        time_text = time_raw.dt.strftime("%H:%M")
    else:
        # Otherwise treat it as text. Trim to the first five characters so
        # '08:42:00' and '08:42' both work.
        time_text = time_raw.astype(str).str.strip().str[:5]

    combined = date_text + " " + time_text
    frame["event_time"] = pd.to_datetime(combined, errors="coerce")

    n_bad = int(frame["event_time"].isna().sum())
    if n_bad:
        print(f"  note: {n_bad:,} rows had an unparseable date or time")

    return frame


def check_parquet_engine():
    """Fail early and clearly if pyarrow is missing.

    pandas needs pyarrow (or fastparquet) to write Parquet, and its own
    error message only appears after all the work is done, which is a
    miserable way to lose five minutes.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("pyarrow is not installed, so this script cannot write Parquet.")
        print("Fix it with:  pip install pyarrow")
        sys.exit(1)


def main():
    check_parquet_engine()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in RAW_DIR.iterdir() if is_delay_file(p))
    if not paths:
        print(f"No delay files found in {RAW_DIR}.")
        print("Run python src/01_download.py first, or put the files there by hand.")
        sys.exit(1)

    print(f"Found {len(paths)} delay files.\n")
    print("ROW ACCOUNTING")
    print("-" * 70)

    frames = []
    for path in paths:
        frame = load_one_file(path)
        if frame is None:
            continue
        log_stage(f"read {path.name}", len(frame))
        frames.append(frame)

    if not frames:
        print("\nNothing could be read. Stopping.")
        sys.exit(1)

    # ignore_index=True renumbers the rows 0..n instead of keeping each
    # file's own numbering, which would give us duplicate index values.
    data = pd.concat(frames, ignore_index=True)
    log_stage("TOTAL read", len(data))

    check_schema(data)

    # ---- types -----------------------------------------------------------
    data = build_event_time(data)

    # errors='coerce' turns anything non-numeric into NaN instead of raising.
    for column in ("min_delay", "min_gap"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Station names: uppercase and strip whitespace. This collapses
    # 'Bloor-Yonge ' and 'BLOOR-YONGE' into one station. It does not fix
    # 'BLOOR-YONGE STATION' vs 'BLOOR-YONGE' - full canonicalisation is a
    # rabbit hole, and Day 1 is not the day for it.
    stations_before = data["station"].nunique()
    data["station"] = data["station"].astype(str).str.strip().str.upper()
    stations_after = data["station"].nunique()
    print(f"  distinct stations: {stations_before:,} -> {stations_after:,}")

    data["line"] = data["line"].astype(str).str.strip().str.upper()
    data["code"] = data["code"].astype(str).str.strip().str.upper()

    # ---- filters, each one counted --------------------------------------
    before = len(data)
    data = data[data["event_time"].notna()]
    log_stage("after dropping unparseable times", len(data), f"-{before - len(data):,}")

    if START_DATE:
        before = len(data)
        data = data[data["event_time"] >= pd.Timestamp(START_DATE)]
        log_stage(f"after start date {START_DATE}", len(data), f"-{before - len(data):,}")

    if END_DATE:
        before = len(data)
        data = data[data["event_time"] <= pd.Timestamp(END_DATE)]
        log_stage(f"after end date {END_DATE}", len(data), f"-{before - len(data):,}")

    # QUALITY CHECK 2 of 4: impossible values.
    before = len(data)
    data = data[data["min_delay"].notna()]
    data = data[data["min_delay"] >= 0]
    data = data[data["min_delay"] <= MAX_PLAUSIBLE_DELAY_MIN]
    log_stage("after range check on min_delay", len(data), f"-{before - len(data):,}")

    # QUALITY CHECK 3 of 4: exact duplicate incidents.
    before = len(data)
    data = data.drop_duplicates(subset=["event_time", "station", "code", "min_delay"])
    log_stage("after dropping duplicates", len(data), f"-{before - len(data):,}")

    # ---- derived columns -------------------------------------------------
    # Added here so every downstream script gets them for free.
    data["hour"] = data["event_time"].dt.hour
    data["dayofweek"] = data["event_time"].dt.dayofweek  # Monday = 0
    data["is_weekend"] = data["dayofweek"] >= 5
    data["month"] = data["event_time"].dt.to_period("M").astype(str)

    # Note we are NOT filtering to min_delay >= 1 here. The Parquet is the
    # clean record of what the TTC published. Filtering to actual delays is
    # a modelling decision and belongs in the modelling script.
    n_actual = int((data["min_delay"] >= 1).sum())

    data = data.sort_values("event_time").reset_index(drop=True)

    # ---- write -----------------------------------------------------------
    data.to_parquet(DELAYS_PARQUET, index=False)

    print("-" * 70)
    print(f"\nWrote {len(data):,} rows to {DELAYS_PARQUET}")
    print(f"Date range : {data['event_time'].min()} to {data['event_time'].max()}")
    print(f"Rows with min_delay >= 1 : {n_actual:,} ({n_actual / len(data):.1%})")
    print(f"Distinct stations : {data['station'].nunique():,}")
    print(f"Distinct codes    : {data['code'].nunique():,}")
    print("\nPut the final row count and date range in docs/log.md.")


if __name__ == "__main__":
    main()
