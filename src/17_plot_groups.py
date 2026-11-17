"""
Draw the Day 4 audit figure from the saved audit.

Run it with:
    python src/17_plot_groups.py

Writes docs/figures/day4_group_coverage.png. Reads docs/day4_audit.json
only, so it can be re-run and restyled freely: it never touches the test set.

THE FIGURE
----------
One panel per grouping (line, hour band, code tier), one row per group.
The dot is test coverage; the thick bar its 95% interval; the thin bar the
interval adjusted for testing every group at once, which is what the flag
rule uses. A group undercovers if its thin bar ends left of the dashed 90%
line. The dotted line is overall coverage. Hollow dots are groups under 100
calibration rows: shown, never flagged.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")  # write a file; do not try to open a window
import matplotlib.pyplot as plt

from config import DAY4_AUDIT, DOCS_DIR

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"

TITLES = {"line": "Line", "hour_band": "Hour band", "code_tier": "Delay code, by its typical length"}
FLAG_TEXT = {"undercovers": "undercovers", "possible": "possible", "ok": "", "description only": "thin"}


def main():
    if not DAY4_AUDIT.exists():
        print("No audit yet. Run: python src/16_audit.py")
        return 1
    record = json.loads(DAY4_AUDIT.read_text())
    target = record["target_coverage"] * 100
    overall = record["overall"] * 100
    groupings = list(dict.fromkeys(r["grouping"] for r in record["rows"]))
    by = {g: [r for r in record["rows"] if r["grouping"] == g] for g in groupings}

    lows = [r["ci_adj_low"] * 100 for r in record["rows"]]
    highs = [r["ci_adj_high"] * 100 for r in record["rows"]]
    x_lo = min(min(lows), target) - 2
    x_hi = max(max(highs), overall) + 2

    fig, axes = plt.subplots(
        len(groupings), 1, figsize=(9, 1.2 + 0.42 * len(record["rows"]) + 0.5 * len(groupings)),
        gridspec_kw={"height_ratios": [len(by[g]) + 1 for g in groupings]},
        sharex=True, facecolor=SURFACE,
    )
    for ax, g in zip(axes, groupings):
        rows = by[g]
        ys = list(range(len(rows)))[::-1]  # first group at the top
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axvline(target, color=INK_SOFT, linewidth=1, linestyle=(0, (4, 3)))
        ax.axvline(overall, color=INK_SOFT, linewidth=1, linestyle=(0, (1, 2)))
        for y, r in zip(ys, rows):
            ax.hlines(y, r["ci_adj_low"] * 100, r["ci_adj_high"] * 100, color=SERIES,
                      linewidth=1.2, alpha=0.45)
            ax.hlines(y, r["ci95_low"] * 100, r["ci95_high"] * 100, color=SERIES, linewidth=3.2)
            thin = r["flag"] == "description only"
            ax.plot(r["coverage"] * 100, y, "o", markersize=8,
                    markerfacecolor=SURFACE if thin else SERIES,
                    markeredgecolor=SERIES if thin else SURFACE, markeredgewidth=1.5)
            note = FLAG_TEXT[r["flag"]]
            if note:
                ax.text(x_hi - 0.2, y, note, ha="right", va="center", fontsize=9,
                        color=INK if r["flag"] == "undercovers" else INK_SOFT,
                        fontweight="bold" if r["flag"] == "undercovers" else "normal")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{r['group']}  ({r['n']:,})" for r in rows], color=INK, fontsize=9)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title(TITLES.get(g, g), color=INK, fontsize=10, loc="left")
    axes[-1].set_xlim(x_lo, x_hi)
    axes[-1].set_xlabel("Test-set coverage (%)", color=INK_SOFT, fontsize=9)
    # Label the two reference lines outward from each other so they never
    # overlap, whichever side of the target overall coverage lands.
    top, y_label = axes[0], len(by[groupings[0]]) - 0.35
    left_first = overall >= target
    top.text(target, y_label, f"{target:.0f}% target " if left_first else f" {target:.0f}% target",
             color=INK_SOFT, fontsize=8, va="bottom", ha="right" if left_first else "left")
    top.text(overall, y_label, f" overall {overall:.1f}%" if left_first else f"overall {overall:.1f}% ",
             color=INK_SOFT, fontsize=8, va="bottom", ha="left" if left_first else "right")

    fig.suptitle("Coverage inside each group, test set", color=INK, fontsize=12,
                 x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.005,
             f"{record['n_test']:,} test delays (count in brackets). Thick bar: 95% interval; thin bar: "
             f"adjusted for {record['family_size']} groups, used for the flag. Intervals resample whole days.",
             color=INK_SOFT, fontsize=7.5, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    out = DOCS_DIR / "figures" / "day4_group_coverage.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"Wrote {out.relative_to(DOCS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
