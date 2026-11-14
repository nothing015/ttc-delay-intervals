"""
Work out the train / calibration / test split dates from the real data.

Run it with:
    python src/04_splits.py

Prints the two dates to paste into config.py, then checks them.

WHY A SCRIPT AND NOT JUST PICKING DATES
---------------------------------------
Splitting 60/20/20 by *calendar time* only gives you 60/20/20 of the
*rows* if incidents are spread evenly across the period. They are not:
the TTC logs more incidents in some years than others. So we pick the
dates by row count instead, then round them to a month boundary so they
are readable and defensible in a write-up.

Rounding to a month shifts the proportions slightly. The script prints
what you actually end up with so you can see the real numbers rather
than the ones you intended.
"""

import sys

import pandas as pd

from config import CALIB_END, DELAYS_PARQUET, MIN_DELAY_THRESHOLD, TRAIN_END

# The shares we are aiming for.
TRAIN_SHARE = 0.60
CALIB_SHARE = 0.20


def month_floor(timestamp):
    """Round a timestamp down to the first day of its month."""
    return pd.Timestamp(year=timestamp.year, month=timestamp.month, day=1)


def describe(data, name, mask):
    """Print one slice's row count, share and date range."""
    slice_ = data[mask]
    if len(slice_) == 0:
        print(f"  {name:<13} EMPTY  <- this will break everything downstream")
        return
    share = len(slice_) / len(data)
    print(
        f"  {name:<13} {len(slice_):>7,} rows  ({share:5.1%})  "
        f"{slice_['event_time'].min().date()} to {slice_['event_time'].max().date()}"
    )


def main():
    data = pd.read_parquet(DELAYS_PARQUET)

    # Split on the MODELLING population, not every row. The zero-delay rows
    # are in the Parquet as a record of what the TTC published, but they are
    # not what we predict, so they should not influence where the cuts fall.
    data = data[data["min_delay"] >= MIN_DELAY_THRESHOLD].copy()
    data = data.sort_values("event_time")

    print(f"Modelling population: {len(data):,} rows "
          f"({data['event_time'].min().date()} to {data['event_time'].max().date()})\n")

    # The timestamp at the 60th and 80th percentile of rows.
    train_cut_raw = data["event_time"].quantile(TRAIN_SHARE)
    calib_cut_raw = data["event_time"].quantile(TRAIN_SHARE + CALIB_SHARE)

    # Round to month boundaries so the dates are clean to state.
    train_cut = month_floor(train_cut_raw)
    calib_cut = month_floor(calib_cut_raw)

    print("SUGGESTED SPLIT DATES")
    print("-" * 70)
    print(f'TRAIN_END = "{train_cut.date()}"')
    print(f'CALIB_END = "{calib_cut.date()}"')
    print()
    print("What those give you:")
    t = data["event_time"]
    describe(data, "train", t < train_cut)
    describe(data, "calibration", (t >= train_cut) & (t < calib_cut))
    describe(data, "test", t >= calib_cut)

    # ---- what is currently in config.py ---------------------------------
    print("\nWHAT config.py SAYS RIGHT NOW")
    print("-" * 70)
    print(f'TRAIN_END = "{TRAIN_END}"   CALIB_END = "{CALIB_END}"')
    describe(data, "train", t < pd.Timestamp(TRAIN_END))
    describe(data, "calibration",
             (t >= pd.Timestamp(TRAIN_END)) & (t < pd.Timestamp(CALIB_END)))
    describe(data, "test", t >= pd.Timestamp(CALIB_END))

    # ---- is the calibration set big enough? ------------------------------
    n_calib = int(((t >= train_cut) & (t < calib_cut)).sum())
    print("\nIS CALIBRATION BIG ENOUGH?")
    print("-" * 70)
    print(f"  {n_calib:,} calibration rows at the suggested dates")
    print("  Day 3 (one global quantile) needs a few hundred. Fine.")
    print("  Day 4 (a quantile per group) needs a few hundred PER GROUP.")

    if n_calib < 1000:
        print("\n  WARNING: under 1,000 rows. Shift TRAIN_END earlier.")
        sys.exit(1)

    # Rough per-group feasibility check for the Day 4 audit.
    calib = data[(t >= train_cut) & (t < calib_cut)]
    for grouping in ("line", "hour"):
        if grouping not in calib.columns:
            continue
        counts = calib.groupby(grouping).size()
        thin = int((counts < 100).sum())
        print(f"\n  by {grouping}: {len(counts)} groups, "
              f"smallest {counts.min():,}, {thin} under 100 rows")
        if thin:
            print(f"    -> those {thin} will need merging on Day 4")

    print("\nPaste the two suggested dates into src/config.py, then re-run this")
    print("script to confirm the slices look as expected.")


if __name__ == "__main__":
    main()