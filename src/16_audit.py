"""
The Day 4 coverage audit: test coverage inside each group, from the file
Day 3 saved. Runs once.

Run it with:
    python src/16_audit.py

Reads data/processed/day3_test_rows.parquet and docs/day4_groups.json.
Writes docs/day4_audit.json (the lock) and docs/day4_audit.csv.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is not a new measurement of the test set. Day 3 measured every test
delay once and saved whether each was covered. This script only slices that
saved result by groups fixed before it runs: the three groupings from Day 1
and the code tiers from 14_define_groups.py. No model is loaded, nothing is
predicted, and no q_hat is changed.

It still runs once, for the same reason as Day 3: if the groups could be
redrawn and the audit re-run, I could slice until something broke. Once
docs/day4_audit.json exists, the script prints the stored result and stops.

THE RULE, FIXED BEFORE THE RUN
------------------------------
A group UNDERCOVERS if its Bonferroni-adjusted interval (family-wise 95%
over every flaggable group) lies entirely below 90%. Intervals come from
resampling whole days of the test period (coverage_stats.py explains why).
"possible" marks a group whose ordinary 95% interval is below 90% but whose
adjusted one is not: worth naming, not a finding. Groups under 100
calibration rows are shown but never flagged.
"""

import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    AUDIT_BOOTSTRAP_REPS,
    AUDIT_SEED,
    CALIB_END,
    DAY3_TEST_RESULTS,
    DAY3_TEST_ROWS,
    DAY4_AUDIT,
    DAY4_AUDIT_TABLE,
    DAY4_GROUPS,
    DOCS_DIR,
    TARGET_COVERAGE,
)
from coverage_stats import bonferroni_z, day_bootstrap, day_weights, self_test, summarise
from groups import add_groups, tested_groups

AUDIT_RULE = ("a group undercovers if its Bonferroni-adjusted interval (family-wise 95%, "
              "day bootstrap) lies entirely below 90%")


def show(record):
    target = record["target_coverage"]
    print(f"  {record['n_test']:,} test delays, overall coverage {record['overall']:.1%}; "
          f"{record['family_size']} flaggable groups, adjusted z = {record['z_adjusted']:.2f}")
    current = None
    for r in record["rows"]:
        if r["grouping"] != current:
            current = r["grouping"]
            print(f"\n  {current}")
            print(f"    {'group':<24} {'n':>6} {'coverage':>9} {'95% CI':>15} "
                  f"{'adjusted CI':>15} {'deff':>5}  flag")
        flag = {"undercovers": "UNDERCOVERS", "possible": "possible", "ok": "",
                "description only": "(description only)"}[r["flag"]]
        print(f"    {r['group']:<24} {r['n']:>6,} {r['coverage']:>9.1%} "
              f"{r['ci95_low']:>7.1%}-{r['ci95_high']:<6.1%} "
              f"{r['ci_adj_low']:>7.1%}-{r['ci_adj_high']:<6.1%} {r['design_effect']:>5.1f}  {flag}")
    flagged = [r for r in record["rows"] if r["flag"] == "undercovers"]
    possible = [r for r in record["rows"] if r["flag"] == "possible"]
    print()
    if flagged:
        print(f"  {len(flagged)} group(s) undercover the {target:.0%} target under the rule fixed in advance:")
        for r in flagged:
            print(f"    {r['grouping']}: {r['group']}  {r['coverage']:.1%} "
                  f"(adjusted CI up to {r['ci_adj_high']:.1%})")
    else:
        print(f"  No group undercovers the {target:.0%} target under the rule fixed in advance.")
        print("  That is a negative result, and it is reported as one.")
    if possible:
        print("  Possible, not established (95% CI below 90%, adjusted CI not): "
              + ", ".join(f"{r['group']} ({r['coverage']:.1%})" for r in possible))
    for grouping, spread in record["spread"].items():
        print(f"  spread across {grouping}: {spread:.1%} (highest minus lowest coverage)")


def main():
    if DAY4_AUDIT.exists():
        record = json.loads(DAY4_AUDIT.read_text())
        print("THE AUDIT HAS ALREADY RUN")
        print("-" * 70)
        print(f"  run at {record['run_at']}")
        show(record)
        print("\nIt will not run again: the groups and the rule were fixed before it ran.")
        return 1

    print("1. SELF-TEST  (made-up numbers, known answers)")
    print("-" * 70)
    failed = False
    for name, passed, detail in self_test():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + ("" if passed else f"  ({detail})"))
        failed |= not passed
    if failed:
        print("\nThe interval code is wrong. Stop: nothing below can be trusted.")
        return 1

    for path, how in ((DAY3_TEST_RESULTS, "python src/12_test_once.py --confirm"),
                      (DAY3_TEST_ROWS, "python src/12_test_once.py --confirm"),
                      (DAY4_GROUPS, "python src/14_define_groups.py")):
        if not path.exists():
            print(f"\n  Missing {path.name}. Run: {how}")
            return 1
    result = json.loads(DAY3_TEST_RESULTS.read_text())
    spec = json.loads(DAY4_GROUPS.read_text())
    rows = pd.read_parquet(DAY3_TEST_ROWS)

    # The per-row file must be exactly what Day 3 measured.
    assert len(rows) == result["n_test"], "per-row file does not match Day 3's row count"
    assert np.isclose(rows["covered"].mean(), result["coverage"]), "per-row file does not match Day 3's coverage"
    assert rows["event_time"].min() >= pd.Timestamp(CALIB_END), "per-row file holds non-test rows"

    groups = add_groups(rows, spec)
    # The tiers and the model's OTHER category must agree row by row.
    other = groups["code_tier"] == spec["other_label"]
    if not (other == (rows["code_grp"].astype("string") == "OTHER")).all():
        print("  [FAIL] code tiers disagree with the model's OTHER category. Re-run 14_define_groups.py.")
        return 1

    print("\n2. THE RULE, FIXED BEFORE THE RUN")
    print("-" * 70)
    print(f"  {AUDIT_RULE}")
    family = sum(len(tested_groups(spec, g)) for g in spec["order"])
    z95, z_adj = bonferroni_z(1), bonferroni_z(family)
    print(f"  {family} flaggable groups -> each tested at 0.05 / {family}, z = {z_adj:.2f}")

    covered = rows["covered"].to_numpy(dtype=bool)
    day_index, days = pd.factorize(rows["event_time"].dt.normalize(), sort=True)
    weights = day_weights(len(days), AUDIT_BOOTSTRAP_REPS, AUDIT_SEED)
    no_model = float(result["no_model_bound"])
    upper = rows["upper"].to_numpy(dtype=float)

    out = []
    spread = {}
    for grouping in spec["order"]:
        order = [g for g in spec["order"][grouping] if (groups[grouping] == g).any()]
        flaggable = set(tested_groups(spec, grouping))
        group_index = pd.Categorical(groups[grouping], categories=order).codes
        boot = day_bootstrap(day_index, group_index, covered, len(order), weights)
        coverages = []
        for i, g in enumerate(order):
            m = group_index == i
            stats = summarise(int(covered[m].sum()), int(m.sum()), boot[:, i], z95, z_adj)
            if g not in flaggable:
                flag = "description only"
            elif stats["ci_adj_high"] < TARGET_COVERAGE:
                flag = "undercovers"
            elif stats["ci95_high"] < TARGET_COVERAGE:
                flag = "possible"
            else:
                flag = "ok"
            if g in flaggable:
                coverages.append(stats["coverage"])
            out.append({"grouping": grouping, "group": g, "n": int(m.sum()),
                        "calibration_n": spec["calibration_counts"][grouping].get(g, 0),
                        "covered": int(covered[m].sum()), **stats, "flag": flag,
                        "upper_median": float(np.median(upper[m])),
                        "share_tighter_than_no_model": float(np.mean(upper[m] < no_model))})
        spread[grouping] = max(coverages) - min(coverages) if coverages else float("nan")

    record = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "rule": AUDIT_RULE,
        "target_coverage": TARGET_COVERAGE,
        "score": result["score"],
        "n_test": int(len(rows)),
        "n_days": int(len(days)),
        "overall": float(covered.mean()),
        "family_size": int(family),
        "z_95": z95,
        "z_adjusted": z_adj,
        "bootstrap_reps": AUDIT_BOOTSTRAP_REPS,
        "seed": AUDIT_SEED,
        "groups_file": DAY4_GROUPS.name,
        "spread": spread,
        "rows": out,
    }

    print("\n3. COVERAGE BY GROUP  (test set, from Day 3's per-row file)")
    print("-" * 70)
    show(record)
    print("\n  deff = design effect: how much wider the day bootstrap is than a")
    print("  count of independent rows would suggest (1 = no clustering by day).")
    print("  This is still the Day 3 measurement, sliced. Nothing was re-measured,")
    print("  and nothing will be changed because of it. Mondrian stays on calibration.")

    pd.DataFrame(out).to_csv(DAY4_AUDIT_TABLE, index=False)
    DOCS_DIR.mkdir(exist_ok=True)
    DAY4_AUDIT.write_text(json.dumps(record, indent=2))  # last: this file is the lock
    print(f"\nWrote {DAY4_AUDIT_TABLE.relative_to(DOCS_DIR.parent)}")
    print(f"Wrote {DAY4_AUDIT.relative_to(DOCS_DIR.parent)}")
    print("Next: python src/17_plot_groups.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
