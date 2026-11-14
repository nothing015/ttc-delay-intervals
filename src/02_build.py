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

import re
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


def is_real_excel(path):
    """True if the file really is an Excel workbook, judged by its contents.

    An .xlsx file is a zip archive, and every zip archive starts with the two
    bytes 'PK'. The portal serves at least one file named .xlsx that is really
    plain CSV text, so the extension cannot be trusted. pandas refuses to open
    that file with "Excel file format cannot be determined".
    """
    with open(path, "rb") as f:
        return f.read(2) == b"PK"


def years_in_name(path):
    """Four-digit years appearing in a filename: '...-jan-2014-april-2017' -> [2014, 2017]."""
    return [int(y) for y in re.findall(r"(?:19|20)\d\d", path.stem)]


def pick_delay_files():
    """Choose which files in data/raw/ to read, and say why for every skip.

    Three rules, each printed when it fires so nothing disappears quietly:
      1. The readme and the delay-code key are not delay data.
      2. If a file's name says its data is entirely before START_DATE, don't
         read it. We would only throw every row away at the date filter.
      3. If both X.xlsx and X.csv exist, they hold the same data. Use the CSV.
    """
    candidates = [
        p
        for p in sorted(RAW_DIR.iterdir())
        if p.suffix.lower() in {".xlsx", ".xls", ".csv"}
        and "readme" not in p.stem.lower()
        and "code" not in p.stem.lower()
    ]
    csv_stems = {p.stem.lower() for p in candidates if p.suffix.lower() == ".csv"}
    start_year = int(START_DATE[:4]) if START_DATE else None

    chosen = []
    for path in candidates:
        years = years_in_name(path)
        if start_year and years and max(years) < start_year:
            print(f"  skip {path.name}: name says {max(years)} or earlier, before START_DATE")
            continue
        if path.suffix.lower() != ".csv" and path.stem.lower() in csv_stems:
            print(f"  skip {path.name}: a .csv copy of the same data exists, using that")
            continue
        chosen.append(path)
    return chosen


def read_raw(path):
    """Read one raw file into a DataFrame, whatever it actually is."""
    # CSV text, or a file named Excel that is really CSV text.
    if path.suffix.lower() == ".csv" or not is_real_excel(path):
        if path.suffix.lower() != ".csv":
            print(f"    note: {path.name} is named Excel but is plain text; reading as CSV")
        # utf-8-sig drops an invisible byte-order mark some exports start with.
        # 'replace' means a stray odd character in a station name cannot crash
        # the whole build.
        return pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace")

    workbook = pd.ExcelFile(path)
    if len(workbook.sheet_names) > 1:
        print(
            f"    !! {path.name} has {len(workbook.sheet_names)} sheets. "
            f"Reading ONLY the first; rows on the other sheets are NOT included."
        )
    # Passing the open ExcelFile avoids re-reading the file from disk.
    return pd.read_excel(workbook, sheet_name=0)


def load_one_file(path):
    """Read one raw file and rename its columns to the canonical set.

    Returns a DataFrame, or None if the file could not be used.
    """
    try:
        frame = read_raw(path)
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

    # Step 4: check THIS file has the columns we need, before it is stacked
    # with the others. Checking after stacking is useless: if 2023 is missing
    # min_delay but 2024 has it, the combined table has the column and the
    # check passes, while every 2023 row silently becomes null and is thrown
    # away later.
    check_schema(frame, path.name)

    # Step 5: build the timestamp NOW, one file at a time, never after the
    frame = build_event_time(frame, path.name)

    frame["source_file"] = path.name
    return frame


def check_schema(frame, name):
    """Stop the build if a required column is missing from THIS file.

    QUALITY CHECK 1 of 4. Deliberately fatal: a missing column means the
    file format changed, and quietly carrying on would produce a Parquet
    that looks fine and is wrong.

    Runs per file, never on the stacked table. A column missing from one
    year is hidden once the years are concatenated, because the other
    years supply it and the rows from the odd one out just fill with
    nulls.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        print(f"\n{'=' * 70}")
        print(f"SCHEMA CHECK FAILED in {name}")
        print(f"{'=' * 70}")
        print(f"  missing  : {missing}")
        print(f"  present  : {sorted(c for c in frame.columns if c != 'source_file')}")
        print("\nOne of the present columns is probably the missing one under a")
        print("new name. Add that spelling to COLUMN_ALIASES in src/config.py,")
        print("mapping it to the canonical name, then run this script again.")
        sys.exit(1)


def build_event_time(frame, name):
    """Combine the separate date and time columns into one timestamp.

    Called once per file. This is the fiddliest part of the whole script. The
    TTC stores the date in one column and the time in another, and the time
    arrives in at least three different shapes depending on the file:

      - a string like '08:42'
      - a datetime where only the time part is meaningful
      - an Excel serial fraction (0.3625 == 08:42)

    Strategy: convert both to strings, glue them together, let pandas parse
    the result. Anything that fails becomes NaT (not-a-time) rather than
    crashing, and we count the failures.
    """
    if "date" not in frame.columns or "time" not in frame.columns:
        print(f"    !! {name} has no date or time column; every row from it will be dropped")
        frame["event_time"] = pd.NaT
        return frame

    date_text = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    time_raw = frame["time"]

    if pd.api.types.is_datetime64_any_dtype(time_raw):
        # A datetime where only the time part matters.
        time_text = time_raw.dt.strftime("%H:%M")
    elif pd.api.types.is_numeric_dtype(time_raw):
        # Excel stores a time of day as a fraction of 24 hours: 0.5 == 12:00.
        offset = pd.to_timedelta(time_raw, unit="D")
        time_text = (pd.Timestamp("2000-01-01") + offset).dt.strftime("%H:%M")
    else:
        # Text. Trim to the first five characters so '08:42:00' and '08:42'
        # both work.
        time_text = time_raw.astype(str).str.strip().str[:5]

    combined = date_text + " " + time_text
    frame["event_time"] = pd.to_datetime(combined, errors="coerce")

    n_bad = int(frame["event_time"].isna().sum())
    if n_bad:
        print(f"    note: {name}: {n_bad:,} rows had an unparseable date or time")

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

    print("FILE SELECTION")
    print("-" * 70)
    paths = pick_delay_files()
    if not paths:
        print(f"No delay files found in {RAW_DIR}.")
        print("Run python src/01_download.py first, or put the files there by hand.")
        sys.exit(1)

    print(f"\nReading {len(paths)} files.\n")
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

    print(f"  schema check: passed for all {len(frames)} files")

    # ---- types -----------------------------------------------------------
    # (event_time was already built per file inside load_one_file.)

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
    data["dayofweek"] = data["event_time"].dt.day_of_week  # Monday = 0
    data["is_weekend"] = data["dayofweek"] >= 5
    data["month"] = data["event_time"].dt.to_period("M").astype(str)

    # Note we are NOT filtering to min_delay >= 1 here. The Parquet is the
    # clean record of what the TTC published. Filtering to actual delays is
    # a modelling decision and belongs in the modelling script.
    n_actual = int((data["min_delay"] >= 1).sum())

    data = data.sort_values("event_time").reset_index(drop=True)

    # The raw date and time columns are fully replaced by event_time, and
    # their contents differ between Excel and CSV files.
    data = data.drop(columns=["date", "time"])

    # Parquet needs every column to hold ONE type. A column mixing, say, Excel
    # numbers and CSV text cannot be written, so turn any leftover mixed
    # column into text.
    for column in data.columns:
        if data[column].dtype == "object":
            data[column] = data[column].astype("string")

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