"""
Day 1 acceptance test. Asserts everything Day 2 onward depends on.

Run it with:
    python src/07_verify.py

Prints PASS or FAIL per check and exits with status 1 if anything failed, so
it can go in CI later. Read-only: it changes no data.

WHY THIS EXISTS
---------------
Most of these checks would be caught eventually by a wrong-looking number
three days from now. Two of them would not be caught at all.

The first is split integrity. Conformal prediction's coverage guarantee holds
only if the calibration set is never trained on. If the train and calibration
masks overlap by even a few hundred rows, the conformal quantile comes out too
small, the intervals come out too narrow, and the measured coverage still
looks plausible. There is no error message for this. It has to be asserted.

The second is the derived columns. `hour` is computed once in 02_build.py and
read everywhere after. If it ever stops matching event_time, every hour-based
grouping on Day 4 silently groups by the wrong thing.
"""

import sys

import pandas as pd

from config import (
    CALIB_END,
    CODE_KEY_FILE,
    DELAYS_PARQUET,
    HOUR_BANDS,
    LINE_ALIASES,
    MAX_PLAUSIBLE_DELAY_MIN,
    MIN_DELAY_THRESHOLD,
    MIN_ROWS_PER_GROUP,
    MODEL_LINES,
    RAW_DIR,
    REQUIRED_COLUMNS,
    START_DATE,
    TRAIN_END,
)

# 02_build.py consumes the raw `date` and `time` columns into `event_time` and
# then drops them, so the Parquet's schema is NOT REQUIRED_COLUMNS, which
# describes the raw files. These are the columns every later script reads.
PARQUET_COLUMNS = [c for c in REQUIRED_COLUMNS if c not in ("date", "time")] + [
    "event_time",
    "hour",
    "dayofweek",
    "month",
]

RESULTS = []


def check(label, ok, detail=""):
    ok = bool(ok)
    RESULTS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if detail:
        for line in str(detail).splitlines():
            print(f"         {line}")
    return ok


def section(title):
    print(f"\n{title}")
    print("-" * 70)


def main():
    print("DAY 1 ACCEPTANCE TEST")
    print("=" * 70)

    # ------------------------------------------------------------------
    section("1. THE PARQUET")
    # ------------------------------------------------------------------
    if not DELAYS_PARQUET.exists():
        check(f"{DELAYS_PARQUET.name} exists", False,
              "Run python src/02_build.py first. Nothing else can be checked.")
        return finish()
    check(f"{DELAYS_PARQUET.name} exists", True)

    data = pd.read_parquet(DELAYS_PARQUET)
    check("loads and is non-empty", len(data) > 0, f"{len(data):,} rows")

    missing_columns = [c for c in PARQUET_COLUMNS if c not in data.columns]
    check("every column Day 2 onward reads is present", not missing_columns,
          f"missing: {missing_columns}" if missing_columns
          else f"{len(data.columns)} columns, including all "
               f"{len(PARQUET_COLUMNS)} that are read later")
    if missing_columns:
        return finish()

    # Everything below compares timestamps, so if this one fails there is no
    # point continuing: the comparisons would raise TypeError rather than
    # reporting a failure.
    if not check("event_time is a real timestamp column",
                 pd.api.types.is_datetime64_any_dtype(data["event_time"]),
                 f"dtype is {data['event_time'].dtype}; must be datetime64, "
                 f"not object. The build writes it correctly, so a string "
                 f"dtype here usually means the Parquet was written by "
                 f"something else, or read with a different engine."):
        return finish()

    n_null_time = int(data["event_time"].isna().sum())
    check("no null event_time", n_null_time == 0, f"{n_null_time:,} nulls")

    first, last = data["event_time"].min(), data["event_time"].max()
    check("event_time starts at or after START_DATE",
          first >= pd.Timestamp(START_DATE),
          f"{first} vs START_DATE {START_DATE}")
    check("event_time is not in the future",
          last <= pd.Timestamp.now(),
          f"latest row {last}")

    # hour is computed in the build and trusted by every later script.
    hour_matches = (data["hour"] == data["event_time"].dt.hour)
    check("hour column agrees with event_time",
          hour_matches.all(),
          f"{int((~hour_matches).sum()):,} rows disagree")

    delays = data["min_delay"]
    check("min_delay is numeric", pd.api.types.is_numeric_dtype(delays),
          f"dtype is {delays.dtype}")
    check("min_delay has no nulls", int(delays.isna().sum()) == 0,
          f"{int(delays.isna().sum()):,} nulls")
    check("min_delay is non-negative", (delays >= 0).all(),
          f"minimum is {delays.min()}")
    check(f"min_delay within the {MAX_PLAUSIBLE_DELAY_MIN}-minute ceiling",
          (delays <= MAX_PLAUSIBLE_DELAY_MIN).all(),
          f"maximum is {delays.max()}")

    key = ["event_time", "station", "code", "min_delay"]
    n_duplicated = int(data.duplicated(subset=key).sum())
    check("no duplicate incidents", n_duplicated == 0,
          f"{n_duplicated:,} rows duplicate on {key}")

    delayed = data[data["min_delay"] >= MIN_DELAY_THRESHOLD]
    check("modelling population is large enough to model",
          len(delayed) >= 10_000,
          f"{len(delayed):,} rows with min_delay >= {MIN_DELAY_THRESHOLD} "
          f"({len(delayed) / len(data):.1%} of the file)")

    # ------------------------------------------------------------------
    section("2. THE SPLITS  (the ones that silently break coverage)")
    # ------------------------------------------------------------------
    train_end, calib_end = pd.Timestamp(TRAIN_END), pd.Timestamp(CALIB_END)
    check("TRAIN_END is before CALIB_END", train_end < calib_end,
          f"{TRAIN_END} -> {CALIB_END}")

    t = delayed["event_time"]
    is_train = t < train_end
    is_calib = (t >= train_end) & (t < calib_end)
    is_test = t >= calib_end

    n_train, n_calib, n_test = map(int, (is_train.sum(), is_calib.sum(), is_test.sum()))
    total = n_train + n_calib + n_test

    check("the three slices are mutually exclusive",
          int((is_train & is_calib).sum()) == 0
          and int((is_calib & is_test).sum()) == 0
          and int((is_train & is_test).sum()) == 0,
          "an overlap here makes the conformal quantile too small and the "
          "intervals too narrow, with no error raised")
    check("the three slices cover every row", total == len(delayed),
          f"{total:,} of {len(delayed):,} rows assigned")
    check("no slice is empty", min(n_train, n_calib, n_test) > 0,
          f"train {n_train:,} ({n_train / len(delayed):.1%}) | "
          f"calib {n_calib:,} ({n_calib / len(delayed):.1%}) | "
          f"test {n_test:,} ({n_test / len(delayed):.1%})")

    if min(n_train, n_calib, n_test) > 0:
        check("the slices are in chronological order with no overlap",
              t[is_train].max() < t[is_calib].min() < t[is_calib].max() < t[is_test].min(),
              f"train ends {t[is_train].max()}\n"
              f"calib {t[is_calib].min()} to {t[is_calib].max()}\n"
              f"test starts {t[is_test].min()}")

    check("calibration is big enough for a 90% quantile", n_calib >= 1_000,
          f"{n_calib:,} rows; the empirical quantile's error scales like "
          f"1/sqrt(n), so a few hundred is not enough")

    # ------------------------------------------------------------------
    section("3. THE MODELLING POPULATION  (after the line decisions)")
    # ------------------------------------------------------------------
    # Note on dtypes: from pandas 3.0, .astype(str) leaves a missing value
    # missing instead of turning it into the string "nan". So a null line
    # stays null here, isin() puts it in `dropped`, and anything that sorts
    # the labels has to cope with a NaN among the strings.
    lines = delayed["line"].astype("string").str.strip().str.upper()
    lines = lines.replace(LINE_ALIASES)
    keep = lines.isin(MODEL_LINES)
    population = delayed[keep]
    dropped = delayed[~keep]

    check("every remapped line is one we model",
          set(lines[keep].dropna().unique()) <= set(MODEL_LINES)
          and int(lines[keep].isna().sum()) == 0,
          f"kept {sorted(lines[keep].dropna().unique())}")

    dropped_labels = dropped["line"].astype("string").fillna("(missing)").value_counts()
    check("the dropped rows are a small minority",
          len(dropped) / len(delayed) < 0.05,
          f"dropped {len(dropped):,} of {len(delayed):,} "
          f"({len(dropped) / len(delayed):.1%})\n"
          + "\n".join(f"{name:<24} {n:>6,}" for name, n in dropped_labels.items()))

    # A null station or code reaches the population, unlike a null line, and
    # pandas groupby drops null keys silently, so a grouping on Day 4 would
    # quietly lose those rows instead of failing.
    null_counts = {c: int(population[c].isna().sum())
                   for c in ("line", "station", "code")}
    check("no missing line, station or code in the population",
          sum(null_counts.values()) == 0,
          ", ".join(f"{c}: {n:,}" for c, n in null_counts.items()))

    # The reason SRT is dropped rather than modelled: it closed inside the
    # training period, so it cannot affect calibration or test anyway.
    srt = delayed[delayed["line"].astype(str).str.upper() == "SRT"]
    if len(srt):
        check("SRT sits entirely inside the training period",
              srt["event_time"].max() < train_end,
              f"{len(srt):,} SRT rows, last on {srt['event_time'].max().date()}, "
              f"TRAIN_END is {TRAIN_END}")

    # The slice counts in section 2 are over ALL delayed rows. These are the
    # counts Day 2 will actually train, calibrate and test on.
    pop_t = population["event_time"]
    pop_counts = (int((pop_t < train_end).sum()),
                  int(((pop_t >= train_end) & (pop_t < calib_end)).sum()),
                  int((pop_t >= calib_end).sum()))
    check("the modelling population still has all three slices",
          min(pop_counts) > 0,
          f"{len(population):,} rows: train {pop_counts[0]:,} "
          f"({pop_counts[0] / len(population):.1%}) | "
          f"calib {pop_counts[1]:,} ({pop_counts[1] / len(population):.1%}) | "
          f"test {pop_counts[2]:,} ({pop_counts[2] / len(population):.1%})")

    # Day 4 needs every group to support its own quantile.
    calib_population = population[
        (population["event_time"] >= train_end)
        & (population["event_time"] < calib_end)
    ]
    band_of = {h: name for name, hours in HOUR_BANDS for h in hours}
    unassigned = sorted(set(calib_population["hour"].unique()) - set(band_of))
    check("every hour falls in a band", not unassigned,
          f"unassigned hours: {unassigned}")

    band_counts = calib_population["hour"].map(band_of).value_counts()
    thin_bands = band_counts[band_counts < MIN_ROWS_PER_GROUP]
    check(f"every hour band clears {MIN_ROWS_PER_GROUP} calibration rows",
          len(thin_bands) == 0 and len(band_counts) == len(HOUR_BANDS),
          "\n".join(f"{name:<20} {n:>6,}" for name, n in band_counts.items()))

    line_counts = calib_population["line"].replace(LINE_ALIASES).value_counts()
    thin_lines = line_counts[line_counts < MIN_ROWS_PER_GROUP]
    check(f"every modelled line clears {MIN_ROWS_PER_GROUP} calibration rows",
          len(thin_lines) == 0,
          "\n".join(f"{name:<20} {n:>6,}" for name, n in line_counts.items()))

    # ------------------------------------------------------------------
    section("4. THE DELAY-CODE KEY")
    # ------------------------------------------------------------------
    key_path = RAW_DIR / CODE_KEY_FILE
    if check(f"{CODE_KEY_FILE} is in data/raw", key_path.exists()):
        key_frame = pd.read_csv(key_path)
        if check("it has a CODE column", "CODE" in key_frame.columns,
                 f"columns: {list(key_frame.columns)}"):
            key_codes = set(
                key_frame["CODE"].astype("string").dropna().str.strip().str.upper()
            )
            used = set(population["code"].astype("string").dropna().unique())
            undocumented = used - key_codes
            share = population["code"].isin(undocumented).mean()
            undoc_rows = (population.loc[population["code"].isin(undocumented), "code"]
                          .astype("string").value_counts())
            listing = "\n".join(f"{code:<10} {n:>6,} rows" for code, n in undoc_rows.items())
            check("undocumented codes are a known, small share",
                  share < 0.10,
                  f"{len(undocumented)} of {len(used)} codes used in the "
                  f"population are absent from the key, covering {share:.1%} "
                  f"of rows. This is a recorded finding, not a failure.\n"
                  + listing)

    return finish()


def finish():
    failed = [label for label, ok in RESULTS if not ok]
    print("\n" + "=" * 70)
    if failed:
        print(f"{len(failed)} of {len(RESULTS)} checks FAILED:")
        for label in failed:
            print(f"  - {label}")
        print("\nDay 1 is not done. Fix these before starting Day 2.")
        return 1
    print(f"All {len(RESULTS)} checks passed. Day 1 is sound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
