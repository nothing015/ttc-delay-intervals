"""
Answer the eight Day 1 questions and save the plots.

Run it with:
    python src/03_explore.py

Prints answers to the terminal and writes figures to docs/figures/.
Copy the printed answers into docs/log.md.

Each question exists because it changes a decision later in the week.
The comment above each one says which.
"""

import matplotlib

# Use a non-interactive backend so this works over SSH or in a plain
# terminal. Must be set before importing pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DELAYS_PARQUET, DOCS_DIR, MIN_DELAY_THRESHOLD

FIGURES_DIR = DOCS_DIR / "figures"


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"      saved {path.relative_to(DOCS_DIR.parent)}")


def main():
    data = pd.read_parquet(DELAYS_PARQUET)
    print(f"Loaded {len(data):,} rows from {DELAYS_PARQUET.name}\n")

    # "delayed" = the rows we will actually model.
    delayed = data[data["min_delay"] >= MIN_DELAY_THRESHOLD].copy()

    # ----------------------------------------------------------------
    # Q1 - share of zero-delay rows
    # Decides: the modelling population.
    # ----------------------------------------------------------------
    share_zero = (data["min_delay"] < MIN_DELAY_THRESHOLD).mean()
    print("Q1  Share of rows with no real delay")
    print(f"      {share_zero:.1%} of rows are below {MIN_DELAY_THRESHOLD} min")
    print(f"      modelling population: {len(delayed):,} rows\n")

    # ----------------------------------------------------------------
    # Q2 - distribution of positive delays
    # Decides: whether to model on a log scale. The median-to-p90 gap is
    # the number that justifies the whole project, so keep it.
    # ----------------------------------------------------------------
    d = delayed["min_delay"]
    median, p90, p99 = d.median(), d.quantile(0.90), d.quantile(0.99)
    print("Q2  Distribution of actual delays (minutes)")
    print(f"      median {median:.0f} | p90 {p90:.0f} | p99 {p99:.0f} | max {d.max():.0f}")
    print(f"      >>> a rider told '{median:.0f} min' faces up to {p90:.0f} one time in ten")
    print(f"      skew {d.skew():.2f} (log1p skew {np.log1p(d).skew():.2f})\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(d, bins=60)
    axes[0].set_title("Delay (minutes)")
    axes[1].hist(np.log1p(d), bins=60)
    axes[1].set_title("log1p(delay)")
    for ax in axes:
        ax.set_ylabel("incidents")
    save(fig, "q2_delay_distribution")

    # ----------------------------------------------------------------
    # Q3 - rows per month
    # Decides: whether your date range contains gaps or regime shifts.
    # ----------------------------------------------------------------
    by_month = data.groupby("month").size()
    print("Q3  Incidents per month")
    print(f"      {len(by_month)} months, {by_month.min():,} to {by_month.max():,} per month")
    gaps = by_month[by_month < by_month.median() * 0.5]
    print(f"      suspiciously thin months: {list(gaps.index) if len(gaps) else 'none'}\n")

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(range(len(by_month)), by_month.values, marker="o", markersize=3)
    ax.set_xticks(range(0, len(by_month), max(1, len(by_month) // 12)))
    ax.set_xticklabels(
        by_month.index[:: max(1, len(by_month) // 12)], rotation=45, ha="right"
    )
    ax.set_ylabel("incidents")
    ax.set_title("Incidents per month")
    save(fig, "q3_incidents_per_month")

    # ----------------------------------------------------------------
    # Q4 - top delay codes by count and by total minutes
    # Decides: your strongest feature. Note the two rankings differ.
    # ----------------------------------------------------------------
    by_code = delayed.groupby("code")["min_delay"].agg(["size", "sum", "median"])
    print("Q4  Top 10 codes by incident count")
    print(by_code.sort_values("size", ascending=False).head(10).to_string())
    print("\n    Top 10 codes by total delay minutes")
    print(by_code.sort_values("sum", ascending=False).head(10).to_string())
    print()

    # ----------------------------------------------------------------
    # Q5 - codes missing from the official lookup
    # Decides: a documented data-quality finding for Part 4.
    # ----------------------------------------------------------------
    print("Q5  Codes missing from the official key")
    print("      (join the codes file here once you have loaded it)")
    print(f"      distinct codes in the data: {data['code'].nunique()}\n")

    # ----------------------------------------------------------------
    # Q6 - station count distribution
    # Decides: whether group-conditional conformal will have enough rows
    # per station on Day 4. A long thin tail is the thing to watch.
    # ----------------------------------------------------------------
    per_station = delayed.groupby("station").size().sort_values(ascending=False)
    thin = (per_station < 100).sum()
    print("Q6  Stations")
    print(f"      {len(per_station)} distinct stations")
    print(f"      {thin} have fewer than 100 delayed incidents")
    print(f"      median per station: {per_station.median():.0f}")
    print("      >>> thin stations are what will break per-station grouping\n")

    # ----------------------------------------------------------------
    # Q7 and Q8 - median AND spread by hour
    # Q8 is the important one. If the spread varies by hour while the
    # median does not, a single global interval width cannot serve both,
    # which is exactly the Day 4 hypothesis.
    # ----------------------------------------------------------------
    by_hour = delayed.groupby("hour")["min_delay"].agg(
        n="size",
        median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        q90=lambda s: s.quantile(0.90),
    )
    by_hour["iqr"] = by_hour["q75"] - by_hour["q25"]

    print("Q7/Q8  Delay by hour of day")
    print(by_hour.round(1).to_string())

    rush = by_hour.loc[by_hour.index.isin([7, 8, 9, 16, 17, 18])]
    quiet = by_hour.loc[~by_hour.index.isin([7, 8, 9, 16, 17, 18])]
    print(f"\n      rush-hour median {rush['median'].mean():.1f}  IQR {rush['iqr'].mean():.1f}")
    print(f"      off-peak  median {quiet['median'].mean():.1f}  IQR {quiet['iqr'].mean():.1f}")
    ratio = rush["iqr"].mean() / max(quiet["iqr"].mean(), 1e-9)
    print(f"      spread ratio rush/off-peak: {ratio:.2f}x")
    if ratio > 1.3:
        print("      >>> SPREAD VARIES BY HOUR. This is your Day 4 finding,")
        print("          showing up on Day 1. Write a prediction in your log.")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(by_hour.index, by_hour["median"], marker="o")
    axes[0].set_title("Median delay by hour")
    axes[1].plot(by_hour.index, by_hour["iqr"], marker="o", color="crimson")
    axes[1].set_title("Spread (IQR) by hour  <- the one that matters")
    for ax in axes:
        ax.set_xlabel("hour of day")
        ax.set_ylabel("minutes")
        ax.grid(alpha=0.3)
    save(fig, "q8_spread_by_hour")

    print("Copy the printed answers into docs/log.md before you stop.")


if __name__ == "__main__":
    main()
