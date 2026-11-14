"""
Look at three things Day 1 left unanswered. Read-only: changes no data.

Run it with:
    python src/06_peek.py

1. Which values appear in the `line` column, when each first and last
   appears, and how many sit in the calibration slice. A line that stops
   appearing (the SRT closed in 2023) or a one-row label (a typo) should
   not become a conformal group.
2. How much of the data sits in stations with fewer than 100 incidents.
3. What the delay-code key file looks like, so Q5 can be written against
   its real layout instead of a guess.
"""

import pandas as pd

from config import (
    CALIB_END,
    DELAYS_PARQUET,
    MIN_DELAY_THRESHOLD,
    RAW_DIR,
    TRAIN_END,
)

STATION_FLOOR = 100


def peek_lines(delayed):
    print("1. LINE VALUES (delayed rows only)")
    print("-" * 70)
    t = delayed["event_time"]
    in_calib = (t >= pd.Timestamp(TRAIN_END)) & (t < pd.Timestamp(CALIB_END))
    table = delayed.groupby("line").agg(
        rows=("line", "size"),
        first=("event_time", "min"),
        last=("event_time", "max"),
    )
    table["in_calib"] = delayed[in_calib].groupby("line").size()
    table["in_calib"] = table["in_calib"].fillna(0).astype(int)
    table["first"] = table["first"].dt.date
    table["last"] = table["last"].dt.date
    print(table.sort_values("rows", ascending=False).to_string())
    print("\n  Read it like this: a label with a last date well before the end")
    print("  of the data is a closed line; a label with 1-3 rows is a typo or")
    print("  junk. Neither belongs in a conformal group as it stands.\n")


def peek_stations(delayed):
    print("2. STATION TAIL")
    print("-" * 70)
    per_station = delayed.groupby("station").size()
    keep = per_station[per_station >= STATION_FLOOR]
    in_tail = per_station[per_station < STATION_FLOOR].sum()
    print(f"  {len(keep)} stations have >= {STATION_FLOOR} incidents and cover "
          f"{keep.sum():,} rows ({keep.sum() / len(delayed):.1%})")
    print(f"  {len(per_station) - len(keep)} stations have fewer and cover "
          f"{in_tail:,} rows ({in_tail / len(delayed):.1%})")
    print("\n  Top 15 stations:")
    print(per_station.sort_values(ascending=False).head(15).to_string())
    print("\n  A sample of the tail, to see whether it is real stations or noise:")
    tail = per_station[per_station < STATION_FLOOR]
    print(tail.sample(min(15, len(tail)), random_state=0).to_string())
    print()


def peek_code_key():
    print("3. DELAY-CODE KEY FILE")
    print("-" * 70)
    candidates = sorted(p for p in RAW_DIR.iterdir() if "code" in p.stem.lower())
    if not candidates:
        print("  No file with 'code' in its name in data/raw. List the folder")
        print("  and tell me what the key file is called.\n")
        return
    for path in candidates:
        print(f"\n  FILE: {path.name}")
        try:
            if path.suffix.lower() == ".csv":
                sheets = {"(csv)": pd.read_csv(path, header=None, nrows=15)}
            else:
                sheets = pd.read_excel(path, sheet_name=None, header=None, nrows=15)
        except Exception as error:
            print(f"    could not read: {type(error).__name__}: {error}")
            continue
        for name, frame in sheets.items():
            print(f"\n    sheet {name!r}: first 15 rows, no header assumed")
            print(frame.to_string(max_colwidth=40))
    print()


def main():
    data = pd.read_parquet(DELAYS_PARQUET)
    delayed = data[data["min_delay"] >= MIN_DELAY_THRESHOLD]
    peek_lines(delayed)
    peek_stations(delayed)
    peek_code_key()
    print("Paste this whole output back to me.")


if __name__ == "__main__":
    main()
