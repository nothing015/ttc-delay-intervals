"""
Would Mondrian (one q_hat per group) fix group coverage? Checked on the
CALIBRATION slice only.

Run it with:
    python src/15_mondrian.py

Writes docs/day4_mondrian.json. Re-runnable: it never reads a test row, and
its q_hats are never applied to the test set (project rule 2). The script
asserts the first; nothing in the project does the second.

WHAT IT DOES
------------
1. Self-test     Mondrian on made-up numbers with known answers.
2. Load          Day 2's model, the calibration slice and the Day 4 groups.
                 Recomputes the global q_hat and insists it matches Day 3.
3. Temporal      Fit both methods on the EARLIER half of calibration, check
   halves        coverage per group on the LATER half. Global = one q_hat for
                 everyone (what Day 3 used); Mondrian = one per group.
                 This is the closest thing to "would it have helped on new
                 data" that does not touch the test set. It includes the mild
                 drift the Day 3 rehearsal found.
4. Random        The same over 20 random splits. Exchangeable by construction,
   halves        so Mondrian should sit near 90% in EVERY group. Global need not.
5. Full fit      Mondrian's q_hat per group from all 6,563 calibration rows:
                 what it would use. Recorded for the write-up. Not applied.
"""

import json
import sys

import numpy as np
import pandas as pd

from config import (
    CONFORMAL_SCORE,
    DAY3_CALIBRATION,
    DAY4_GROUPS,
    DAY4_MONDRIAN,
    DOCS_DIR,
    TARGET_COVERAGE,
)
from conformal import conformal_quantile, conformity_scores, interval, load_final_model, predict
from data import load_population
from groups import add_groups, tested_groups
from mondrian import mondrian_quantiles, row_quantiles, self_test

SEEDS = range(20)


def rule(title):
    print(f"\n{title}")
    print("-" * 70)


def split_check(scores, preds, labels, first, second, own):
    """Fit global and Mondrian on `first`, report per-group results on `second`."""
    q_glob, _ = conformal_quantile(scores[first], TARGET_COVERAGE)
    table, _ = mondrian_quantiles(scores[first], labels[first], TARGET_COVERAGE, own)
    q_rows = row_quantiles(labels[second], table, q_glob)
    sub = {q: p[second] for q, p in preds.items()}
    _, upper_glob = interval(CONFORMAL_SCORE, sub, q_glob)
    _, upper_mond = interval(CONFORMAL_SCORE, sub, q_rows)
    s2, l2 = scores[second], labels[second]
    out = {}
    for g in dict.fromkeys(l2.tolist()):
        m = l2 == g
        out[g] = {
            "n": int(m.sum()),
            "global": float(np.mean(s2[m] <= q_glob)),
            "mondrian": float(np.mean(s2[m] <= q_rows[m])),
            "upper_global": float(np.median(upper_glob[m])),
            "upper_mondrian": float(np.median(upper_mond[m])),
        }
    return out


def main():
    rule("1. SELF-TEST  (made-up numbers, known answers)")
    failed = False
    for name, passed, detail in self_test(TARGET_COVERAGE):
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + ("" if passed else f"  ({detail})"))
        failed |= not passed
    if failed:
        print("\nThe Mondrian code is wrong. Stop.")
        return 1

    rule("2. LOAD")
    if not DAY4_GROUPS.exists():
        print("  No groups yet. Run: python src/14_define_groups.py")
        return 1
    spec = json.loads(DAY4_GROUPS.read_text())
    calibration = json.loads(DAY3_CALIBRATION.read_text())
    bundle = load_final_model()
    frame = load_population(verbose=False)
    calib = frame[frame["slice"] == "calib"].sort_values("event_time", kind="stable")
    assert set(calib["slice"]) == {"calib"}, "Mondrian must only see calibration rows"

    y = calib["min_delay"].to_numpy(dtype=float)
    # Plain arrays, so the halves below are picked by POSITION. If the model
    # returned pandas Series, preds[q][positions] would look up row LABELS
    # instead, and calib is a time-sorted slice whose labels are not 0..n-1.
    preds = {q: np.asarray(p, dtype=float) for q, p in predict(bundle, calib).items()}
    scores = np.asarray(conformity_scores(CONFORMAL_SCORE, y, preds), dtype=float)
    q_all, _ = conformal_quantile(scores, TARGET_COVERAGE)
    if not np.isclose(q_all, calibration["q_hat"]):
        print(f"  [FAIL] global q_hat is {q_all:.4f}, Day 3 recorded {calibration['q_hat']:.4f}.")
        return 1
    groups = add_groups(calib, spec)
    n = len(calib)
    half = n // 2
    split_date = pd.Timestamp(calib["event_time"].to_numpy()[half])
    print(f"  calibration rows  {n:,}; global q_hat {q_all:.4f} (matches Day 3)")
    print(f"  temporal split    earlier half before {split_date:%Y-%m-%d}, later half after")
    print("  Coverage below is on calibration rows only. The test set is not read.")

    record = {"score": CONFORMAL_SCORE, "target_coverage": TARGET_COVERAGE,
              "n_calibration": n, "global_q_hat": q_all,
              "temporal_split": f"{split_date:%Y-%m-%d}", "groupings": {}}

    for grouping in spec["order"]:
        labels = groups[grouping].to_numpy(dtype=object)
        own = set(tested_groups(spec, grouping))
        order = spec["order"][grouping]

        temporal = split_check(scores, preds, labels, np.arange(half), np.arange(half, n), own)
        randoms = []
        for seed in SEEDS:
            perm = np.random.default_rng(seed).permutation(n)
            randoms.append(split_check(scores, preds, labels, perm[:half], perm[half:], own))
        full, _ = mondrian_quantiles(scores, labels, TARGET_COVERAGE, own)

        rule(f"3-5. {grouping.upper()}  (coverage on calibration; target {TARGET_COVERAGE:.0%})")
        print(f"  {'':<22} {'later half (temporal)':^31}  {'random halves':^15}  {'all calib':>9}")
        print(f"  {'group':<22} {'n':>5} {'global':>7} {'Mondrian':>9} {'bound g/M':>9}  "
              f"{'global':>7} {'Mondr.':>7}  {'q_hat':>9}")
        rows = []
        for g in order:
            if g not in temporal:
                continue
            t = temporal[g]
            rg = float(np.mean([r[g]["global"] for r in randoms if g in r]))
            rm = float(np.mean([r[g]["mondrian"] for r in randoms if g in r]))
            f = full.get(g, {"q_hat": float("nan"), "n": 0, "k": None, "source": "none"})
            tag = "" if f["source"] == "own" else " *"
            print(f"  {g:<22} {t['n']:>5,} {t['global']:>7.1%} {t['mondrian']:>9.1%} "
                  f"{t['upper_global']:>4.1f}/{t['upper_mondrian']:<4.1f}  {rg:>7.1%} {rm:>7.1%}  "
                  f"{f['q_hat']:>+8.2f}{tag}")
            rows.append({"group": g, "temporal_n": t["n"],
                         "temporal_global": t["global"], "temporal_mondrian": t["mondrian"],
                         "temporal_upper_median_global": t["upper_global"],
                         "temporal_upper_median_mondrian": t["upper_mondrian"],
                         "random_global_mean": rg, "random_mondrian_mean": rm,
                         "full_q_hat": f["q_hat"], "full_n": f["n"], "full_k": f["k"],
                         "full_source": f["source"]})
        if any(r["full_source"] != "own" for r in rows):
            print("  * under 100 calibration rows: uses the global q_hat")
        spread_g = max(r["temporal_global"] for r in rows) - min(r["temporal_global"] for r in rows)
        spread_m = max(r["temporal_mondrian"] for r in rows) - min(r["temporal_mondrian"] for r in rows)
        print(f"  spread across groups, later half: global {spread_g:.1%}, Mondrian {spread_m:.1%}")
        record["groupings"][grouping] = rows

    print("\nHow to read it: under random halves Mondrian should sit near 90% in every")
    print("group; if global does too, the groups did not need their own q_hat. The")
    print("temporal columns add the 2025 drift, so neither method is promised 90%")
    print("there. 'bound g/M' is the median upper bound in minutes, global/Mondrian:")
    print("the price of a group's own guarantee. 'q_hat' is each group's own")
    print(f"correction from all calibration rows (score: {CONFORMAL_SCORE}).")
    print("None of this is applied to the test set.")

    DOCS_DIR.mkdir(exist_ok=True)
    DAY4_MONDRIAN.write_text(json.dumps(record, indent=2))
    print(f"\nWrote {DAY4_MONDRIAN.relative_to(DOCS_DIR.parent)}")
    print("Next: write your Day 4 prediction, commit it, then python src/16_audit.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
