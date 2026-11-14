"""
Look at the groups you will use for the Day 4 conditional coverage audit.

Run it with:
    python src/05_groups.py

Prints, for the CALIBRATION slice specifically, how many rows sit in each
line, each hour and each station. Then proposes hour bands that are large
enough to hold a conformal quantile.

WHY THIS EXISTS
---------------
Day 4 computes a separate calibration quantile per group (Mondrian, or
group-conditional, conformal). A group with five rows cannot support a
90% quantile: the estimate is pure noise, and the interval it produces is
meaningless even though the code runs without complaint.

So the groups have to be designed before Day 4, from the calibration
slice, not after. This script is the input to that decision.
"""

import pandas as pd

from config import (
    CALIB_END,
    DELAYS_PARQUET,
    HOUR_BANDS,
    MIN_DELAY_THRESHOLD,
    MIN_ROWS_PER_GROUP,
    TRAIN_END,
)


def band_of(hour):
    for name, hours in HOUR_BANDS:
        if hour in hours:
            return name
    return "unassigned"


def report(frame, column, title):
    counts = frame.groupby(column).size().sort_values(ascending=False)
    thin = counts[counts < MIN_ROWS_PER_GROUP]
    print(f"\n{title}")
    print("-" * 70)
    print(f"  {len(counts)} groups, smallest {counts.min():,}, largest {counts.max():,}")
    if len(counts) <= 30:
        for key, n in counts.items():
            flag = "  <- too thin" if n < MIN_ROWS_PER_GROUP else ""
            print(f"    {str(key):<28} {n:>7,}{flag}")
    else:
        print(f"    top 10 of {len(counts)}:")
        for key, n in counts.head(10).items():
            print(f"    {str(key):<28} {n:>7,}")
        print(f"    ... and {len(counts) - 10:,} more")
    if len(thin):
        print(f"  {len(thin)} group(s) under {MIN_ROWS_PER_GROUP} rows "
              f"({thin.sum():,} rows total, {thin.sum() / len(frame):.2%} of the slice)")
    return counts


def main():
    data = pd.read_parquet(DELAYS_PARQUET)
    data = data[data["min_delay"] >= MIN_DELAY_THRESHOLD]

    t = data["event_time"]
    calib = data[(t >= pd.Timestamp(TRAIN_END)) & (t < pd.Timestamp(CALIB_END))]

    print(f"Calibration slice: {len(calib):,} rows "
          f"({calib['event_time'].min().date()} to {calib['event_time'].max().date()})")
    print(f"A group needs about {MIN_ROWS_PER_GROUP}+ rows to support its own quantile.")

    report(calib, "line", "BY LINE")
    report(calib, "hour", "BY HOUR (raw, 24 groups)")

    # The proposed fix for hours.
    banded = calib.copy()
    banded["hour_band"] = banded["hour"].map(band_of)
    counts = report(banded, "hour_band", "BY HOUR BAND (proposed grouping)")

    print("\nVERDICT")
    print("-" * 70)
    if (counts >= MIN_ROWS_PER_GROUP).all():
        print(f"  Every hour band clears {MIN_ROWS_PER_GROUP} rows. Use hour bands")
        print("  rather than raw hours for the Day 4 group-conditional audit.")
    else:
        bad = counts[counts < MIN_ROWS_PER_GROUP]
        print(f"  Still thin: {list(bad.index)}. Merge these into a neighbour")
        print("  and edit HOUR_BANDS at the top of this file.")

    report(calib, "station", "BY STATION")
    print("\n  Station is free text with a very long tail, so per-station")
    print("  grouping is not viable. Options: group by line instead, or keep")
    print("  the stations that clear the threshold above and pool the rest")
    print("  into OTHER. Decide this on Day 4, not now.")


if __name__ == "__main__":
    main()
