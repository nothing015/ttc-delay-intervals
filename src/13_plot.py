"""
Draw the Day 3 coverage figure from the saved test results.

Run it with:
    python src/13_plot.py

Writes docs/figures/day3_coverage.png. Reads docs/day3_test_results.json
only, so it can be re-run and restyled freely: it never touches the test
set.

THE FIGURE
----------
Left:  coverage measured on the test set against the level asked for, at
       ten levels from 50% to 95%. A valid method sits on the diagonal.
Right: the same points as a gap (measured minus asked for, in percentage
       points) with 95% intervals, because a one-point miss is invisible
       on the left panel.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")  # write a file; do not try to open a window
import matplotlib.pyplot as plt
import numpy as np

from config import DAY3_TEST_RESULTS, DOCS_DIR

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    if not DAY3_TEST_RESULTS.exists():
        print("No test results yet. Run: python src/12_test_once.py --confirm")
        return 1
    result = json.loads(DAY3_TEST_RESULTS.read_text())
    curve = result["curve"]
    level = np.array([c["level"] for c in curve]) * 100
    cover = np.array([c["coverage"] for c in curve]) * 100
    low = np.array([c["ci_low"] for c in curve]) * 100
    high = np.array([c["ci_high"] for c in curve]) * 100
    target = result["target_coverage"] * 100
    at_target = int(np.argmin(np.abs(level - target)))

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)

    style(left)
    left.plot([45, 100], [45, 100], color=INK_SOFT, linewidth=1, linestyle=(0, (4, 3)))
    left.plot(level, cover, color=SERIES, linewidth=2, marker="o", markersize=6,
              markeredgecolor=SURFACE, markeredgewidth=1.5)
    # Both labels sit in the empty triangle under the diagonal.
    left.text(96.5, 50, "dashed: measured = asked for", color=INK_SOFT, fontsize=9,
              ha="right", va="bottom")
    left.annotate(f"{cover[at_target]:.1f}% at the {target:.0f}% target",
                  xy=(level[at_target], cover[at_target]), xytext=(96.5, 62),
                  textcoords="data", ha="right", va="top", color=INK, fontsize=9,
                  arrowprops={"arrowstyle": "-", "color": INK_SOFT, "linewidth": 0.8,
                              "shrinkA": 2, "shrinkB": 5})
    left.set_xlim(47, 98)
    left.set_ylim(47, 98)
    left.set_aspect("equal")
    left.set_xlabel("Coverage asked for (%)", color=INK_SOFT, fontsize=9)
    left.set_ylabel("Coverage measured on test (%)", color=INK_SOFT, fontsize=9)
    left.set_title("Measured against asked for", color=INK, fontsize=10, loc="left")

    style(right)
    gap, gap_low, gap_high = cover - level, low - level, high - level
    right.axhline(0, color=INK_SOFT, linewidth=1, linestyle=(0, (4, 3)))
    right.vlines(level, gap_low, gap_high, color=SERIES, linewidth=1.5, alpha=0.55)
    right.plot(level, gap, color=SERIES, linestyle="none", marker="o", markersize=6,
               markeredgecolor=SURFACE, markeredgewidth=1.5)
    reach = max(2.0, float(np.max(np.abs(np.r_[gap_low, gap_high]))) * 1.15)
    right.set_ylim(-reach, reach)
    right.set_xlim(47, 98)
    right.text(97, reach * 0.93, "covers more than asked", color=INK_SOFT, fontsize=9,
               va="top", ha="right")
    right.text(97, -reach * 0.93, "covers less than asked", color=INK_SOFT, fontsize=9,
               va="bottom", ha="right")
    right.set_xlabel("Coverage asked for (%)", color=INK_SOFT, fontsize=9)
    right.set_ylabel("Measured minus asked for (points)", color=INK_SOFT, fontsize=9)
    right.set_title("The gap, with 95% intervals", color=INK, fontsize=10, loc="left")

    fig.suptitle(
        f"Split conformal coverage on the test set: {result['coverage']:.1%} "
        f"against a {result['target_coverage']:.0%} target",
        color=INK, fontsize=12, x=0.06, ha="left", y=0.98,
    )
    fig.text(0.06, 0.015,
             f"{result['n_test']:,} test delays, {result['test_from']} to {result['test_to']}. "
             f"Calibrated on {result['n_calibration']:,} delays. Score: {result['score']}. "
             "Marginal coverage only.",
             color=INK_SOFT, fontsize=8, ha="left")
    fig.tight_layout(rect=(0.03, 0.05, 1, 0.94))

    out = DOCS_DIR / "figures" / "day3_coverage.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"Wrote {out.relative_to(DOCS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
