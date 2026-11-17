"""
How sure we are about a coverage figure. Imported by 16_audit.py; nothing
here is run directly.

WHY RESAMPLE WHOLE DAYS
-----------------------
A Wilson interval treats every delay as independent. They are not: one bad
day (a signal failure, a weather event) produces many long delays together,
and if the bound misses one of them it tends to miss the others. Rows from
the same day rise and fall together, so the effective sample is smaller
than the row count and a Wilson interval is too narrow.

The day bootstrap copes with that: draw the days of the test period with
replacement, keep every delay on each drawn day, recompute coverage, repeat.
The spread of those coverages is the standard error. Its ratio to the
independent-rows standard error, squared, is the DESIGN EFFECT: 1 means the
days carried no extra noise, 2 means the group has the information of half
as many independent rows.

WHY THE INTERVAL IS NORMAL AROUND THE BOOTSTRAP STANDARD ERROR
--------------------------------------------------------------
The Bonferroni interval needs the 0.2% tails. A percentile bootstrap with a
few thousand draws has only a handful of points that far out, so its edge
jumps around. Using the bootstrap only for the standard error, and the
normal curve for the tails, is steadier.

WHY BONFERRONI
--------------
With 14 groups each tested at 95%, the chance that at least one is flagged
by noise alone is about 1 - 0.95^14 = 51%. Bonferroni tests each group at
0.05 / 14 instead, so the chance of ANY false flag stays at or below 5%.
It is conservative: a real small undercoverage can be missed. That is the
price of not slicing until something breaks.
"""

import numpy as np
from scipy.stats import norm


def wilson(successes, n, z=1.96):
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - spread), float(centre + spread)


def bonferroni_z(m, alpha=0.05):
    """z for a two-sided interval at family-wise level alpha over m groups."""
    return float(norm.ppf(1 - alpha / (2 * m)))


def day_weights(n_days, reps, seed):
    """How many times each day is drawn, in each bootstrap replicate (reps x days)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_days, np.full(n_days, 1 / n_days), size=reps)


def day_bootstrap(day_index, group_index, covered, n_groups, weights):
    """Bootstrap coverage of every group: an array of shape (reps, n_groups).

    day_index    integer day of each row, 0..n_days-1
    group_index  integer group of each row, 0..n_groups-1
    weights      from day_weights(), shared by every grouping
    """
    n_days = weights.shape[1]
    rows = np.zeros((n_days, n_groups))
    hits = np.zeros((n_days, n_groups))
    np.add.at(rows, (day_index, group_index), 1)
    np.add.at(hits, (day_index, group_index), covered.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (weights @ hits) / (weights @ rows)


def summarise(covered_count, n, boot, z95, z_adj):
    """Coverage, its intervals and its design effect for one group."""
    p = covered_count / n
    se = float(np.nanstd(boot, ddof=1))
    se_iid = float(np.sqrt(p * (1 - p) / n)) if 0 < p < 1 else float("nan")
    lo95, hi95 = max(0.0, p - z95 * se), min(1.0, p + z95 * se)
    lo_adj, hi_adj = max(0.0, p - z_adj * se), min(1.0, p + z_adj * se)
    w_lo, w_hi = wilson(covered_count, n)
    return {
        "coverage": p, "se_day_bootstrap": se, "se_independent": se_iid,
        "design_effect": (se / se_iid) ** 2 if se_iid and se_iid > 0 else float("nan"),
        "ci95_low": lo95, "ci95_high": hi95, "ci_adj_low": lo_adj, "ci_adj_high": hi_adj,
        "wilson_low": w_lo, "wilson_high": w_hi,
    }


def self_test():
    """Checks with known answers, on made-up numbers only."""
    results = []
    z1, z14 = bonferroni_z(1), bonferroni_z(14)
    results.append(("Bonferroni z: 1.96 for one group, about 2.91 for 14",
                    abs(z1 - 1.96) < 0.005 and abs(z14 - 2.914) < 0.01,
                    f"got {z1:.3f} and {z14:.3f}"))

    rng = np.random.default_rng(1)

    # One row per day, independent: the day bootstrap must agree with the
    # textbook standard error sqrt(p(1-p)/n).
    n = 2000
    covered = rng.random(n) < 0.9
    boot = day_bootstrap(np.arange(n), np.zeros(n, int), covered, 1, day_weights(n, 2000, 2))
    se, iid = float(np.std(boot, ddof=1)), float(np.sqrt(0.9 * 0.1 / n))
    results.append(("independent rows: bootstrap SE matches sqrt(p(1-p)/n) within 10%",
                    abs(se / iid - 1) < 0.10, f"ratio {se / iid:.3f}"))

    # 200 days x 10 rows, each day covered entirely or not at all: the rows
    # carry the information of 200, not 2,000, so the design effect is ~10.
    days = 200
    day_hit = rng.random(days) < 0.9
    day_index = np.repeat(np.arange(days), 10)
    boot = day_bootstrap(day_index, np.zeros(days * 10, int), day_hit[day_index], 1,
                         day_weights(days, 2000, 3))
    p = day_hit.mean()
    deff = (np.std(boot, ddof=1) / np.sqrt(p * (1 - p) / (days * 10))) ** 2
    results.append(("days covered all-or-nothing, 10 rows each: design effect about 10",
                    7 < deff < 13, f"got {deff:.1f}"))
    return results
