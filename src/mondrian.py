"""
Mondrian (group-conditional) conformal: one q_hat per group instead of one
for everyone. Imported by 15_mondrian.py; nothing here is run directly.

THE IDEA
--------
Split conformal promises 90% averaged over everyone. A group whose delays
the model underestimates can sit well below 90%, hidden by groups above it.
Mondrian runs the same recipe inside each group: the k-th smallest score
among that group's calibration rows, k = ceil((n_group + 1) x 0.9). The
n + 1 argument then holds inside every group, so each group gets its own
90% guarantee, provided rows are exchangeable WITHIN the group.

THE COST
--------
Each q_hat comes from that group's rows alone, so small groups get noisy
(and usually wider) bounds. That is why groups need about 100 calibration
rows (config.MIN_ROWS_PER_GROUP). Thin groups here fall back to the global
q_hat.

ON THIS PROJECT, MONDRIAN NEVER TOUCHES THE TEST SET
----------------------------------------------------
Deciding to use it after seeing the Day 4 audit, then scoring it on test,
would tune the method to the test set (rule 2). So it is fitted and checked
on the calibration slice only, in 15_mondrian.py.
"""

import math

import numpy as np

from conformal import conformal_quantile


def mondrian_quantiles(scores, groups, coverage, own_groups):
    """q_hat for every group: its own if listed in own_groups, else the global one.

    Returns {group: {"q_hat", "n", "k", "source"}} and the global q_hat.
    """
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups, dtype=object)
    q_global, _ = conformal_quantile(scores, coverage)
    out = {}
    for g in dict.fromkeys(groups.tolist()):
        s = scores[groups == g]
        if g in own_groups:
            q, k = conformal_quantile(s, coverage)
            out[g] = {"q_hat": q, "n": int(len(s)), "k": int(k), "source": "own"}
        else:
            out[g] = {"q_hat": q_global, "n": int(len(s)), "k": None, "source": "global"}
    return out, q_global


def row_quantiles(groups, table, fallback):
    """The q_hat each row gets, given a table from mondrian_quantiles."""
    return np.array([table[g]["q_hat"] if g in table else fallback for g in groups], dtype=float)


def self_test(coverage=0.90):
    results = []

    # The hand case from the runbook, at 50% so it can be done on paper.
    scores = [1, 2, 3, 10, 20, 30]
    groups = ["A", "A", "A", "B", "B", "B"]
    table, q_global = mondrian_quantiles(scores, groups, 0.5, {"A", "B"})
    ok = (q_global == 10.0 and table["A"]["q_hat"] == 2.0 and table["B"]["q_hat"] == 20.0
          and table["A"]["k"] == 2 and table["B"]["k"] == 2)
    results.append(("hand case at 50%: global q_hat 10; A gets 2 and B gets 20 (k = 2 each)",
                    ok, f"got global {q_global}, A {table['A']['q_hat']}, B {table['B']['q_hat']}"))

    # Simulation: group A (70% of rows) has small scores, group B (30%) has
    # scores three times larger. One global q_hat covers about 98% of A and
    # 72% of B; Mondrian should cover about 90% of each.
    rng = np.random.default_rng(0)
    worlds, n = 300, 1000
    cover = {"global": {"A": [], "B": []}, "mondrian": {"A": [], "B": []}}
    for _ in range(worlds):
        g = np.where(rng.random(2 * n) < 0.7, "A", "B")
        s = rng.exponential(np.where(g == "A", 1.0, 3.0))
        table, q_glob = mondrian_quantiles(s[:n], g[:n], coverage, {"A", "B"})
        new_s, new_g = s[n:], g[n:]
        q_row = row_quantiles(new_g, table, q_glob)
        for grp in ("A", "B"):
            m = new_g == grp
            cover["global"][grp].append(np.mean(new_s[m] <= q_glob))
            cover["mondrian"][grp].append(np.mean(new_s[m] <= q_row[m]))
    mean = {how: {grp: float(np.mean(v)) for grp, v in d.items()} for how, d in cover.items()}
    results.append(("simulation: one global q_hat leaves group B well below 90%",
                    mean["global"]["B"] < 0.80, f"global covers B {mean['global']['B']:.3f}"))
    results.append(("simulation: Mondrian covers each group within 1 point of 90%",
                    all(abs(mean["mondrian"][grp] - 0.9) < 0.01 for grp in ("A", "B")),
                    f"A {mean['mondrian']['A']:.3f}, B {mean['mondrian']['B']:.3f}"))
    # Sanity: never a non-finite q_hat for groups this size.
    results.append(("no infinite q_hat for groups of 300 or more rows",
                    all(math.isfinite(t["q_hat"]) for t in table.values()), "infinite q_hat"))
    return results
