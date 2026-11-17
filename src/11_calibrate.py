"""
Compute the conformal quantile from the CALIBRATION slice.

Run it with:
    python src/11_calibrate.py

Writes docs/day3_calibration.json. Safe to run as often as you like: it
never reads a test row's delay. The script asserts this.

WHAT IT DOES, IN ORDER
----------------------
1. Self-test       The conformal code on made-up numbers with known answers.
2. Load            Day 2's model and the calibration slice. Checks the model
                   was trained on exactly the training rows.
3. Before          How often the model's own 90th percentile covers the
                   calibration delays, with no correction.
4. Calibrate       Scores, then q_hat, on all calibration rows.
5. Rehearsal       Two checks that use calibration rows only:
                     random halves    calibrate on one half, check the other.
                                      Exchangeable by construction, so this
                                      should give about 90%. If not, the
                                      code is wrong.
                     temporal halves  calibrate on the earlier half, check
                                      the later half. If this is far from
                                      90%, delays drifted inside 2025, and
                                      the 2026 test set may drift too.
                   Neither changes q_hat. They only tell you what to expect.
6. Width           How wide the intervals are on calibration rows, against
                   a bound that ignores the model entirely.
"""

import json
import sys

import numpy as np
import pandas as pd

from config import CONFORMAL_SCORE, DAY3_CALIBRATION, DOCS_DIR, TARGET_COVERAGE, TRAIN_END
from conformal import (
    Q_HIGH,
    conformal_quantile,
    conformity_scores,
    interval,
    is_covered,
    load_final_model,
    predict,
    self_test,
)
from data import load_population

REHEARSAL_SEEDS = range(20)


def rule(title):
    print(f"\n{title}")
    print("-" * 70)


def main():
    rule("1. SELF-TEST  (made-up numbers, known answers)")
    failed = False
    for name, passed, detail in self_test(TARGET_COVERAGE):
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + ("" if passed else f"  ({detail})"))
        failed |= not passed
    if failed:
        print("\nThe conformal code is wrong. Stop: nothing below can be trusted.")
        return 1

    rule("2. LOAD")
    bundle = load_final_model()
    frame = load_population()
    train = frame[frame["slice"].isin(["fit", "valid"])]
    calib = frame[frame["slice"] == "calib"]

    # Rule 1 of the project: the calibration set is never trained on. The
    # model file records how many rows it was fitted on; if that is not the
    # training count, the file is stale or was fitted on something else.
    if bundle["train_rows"] != len(train):
        print(f"  [FAIL] model was fitted on {bundle['train_rows']:,} rows but training "
              f"has {len(train):,}. Re-run 09_fit_final.py.")
        return 1
    assert calib["event_time"].min() >= pd.Timestamp(TRAIN_END), "calibration overlaps training"
    assert set(calib["slice"]) == {"calib"}
    print(f"  model            {bundle['name']}, fitted on {bundle['train_rows']:,} training rows")
    print(f"  calibration      {len(calib):,} rows, "
          f"{calib['event_time'].min():%Y-%m-%d} to {calib['event_time'].max():%Y-%m-%d}")
    print(f"  score            {CONFORMAL_SCORE}")
    print(f"  target coverage  {TARGET_COVERAGE:.0%}")

    y = calib["min_delay"].to_numpy(dtype=float)
    preds = predict(bundle, calib)

    rule("3. BEFORE CONFORMAL  (the model's own 90th percentile, uncorrected)")
    raw = float(np.mean(y <= preds[Q_HIGH]))
    print(f"  share of calibration delays at or below the predicted q90: {raw:.1%}")
    print(f"  (validation gave 88.5% on Day 2; the target is {TARGET_COVERAGE:.0%})")

    rule("4. CALIBRATE")
    scores = conformity_scores(CONFORMAL_SCORE, y, preds)
    q_hat, k = conformal_quantile(scores, TARGET_COVERAGE)
    n = len(scores)
    print(f"  n = {n:,} calibration scores")
    print(f"  k = ceil((n + 1) x {TARGET_COVERAGE}) = {k:,}   (level k / n = {k / n:.4f})")
    unit = "minutes" if CONFORMAL_SCORE == "upper" else "on the log scale"
    print(f"  q_hat = the {k:,}th smallest score = {q_hat:.4f} {unit}")
    if CONFORMAL_SCORE == "upper":
        direction = "raises" if q_hat > 0 else "lowers"
        print(f"  Meaning: conformal {direction} every predicted q90 by {abs(q_hat):.2f} minutes.")
    else:
        print(f"  Meaning: the interval is the predicted median divided and multiplied "
              f"by about {np.exp(q_hat):.2f}.")
    in_sample = float(np.mean(is_covered(CONFORMAL_SCORE, y, preds, q_hat)))
    print(f"  coverage on the calibration rows themselves: {in_sample:.1%} "
          f"(true by construction, not a result)")

    rule("5. REHEARSAL  (calibration rows only; does not change q_hat)")
    half = n // 2

    def half_coverage(first, second):
        q, _ = conformal_quantile(scores[first], TARGET_COVERAGE)
        return float(np.mean(scores[second] <= q))

    randoms = []
    for seed in REHEARSAL_SEEDS:
        order = np.random.default_rng(seed).permutation(n)
        randoms.append(half_coverage(order[:half], order[half:]))
    randoms = np.array(randoms)
    # load_population returns rows in the Parquet's order; sort by time so
    # "first half" really means "earlier".
    by_time = np.argsort(calib["event_time"].to_numpy(), kind="stable")
    temporal = half_coverage(by_time[:half], by_time[half:])
    split_date = calib["event_time"].to_numpy()[by_time[half]]

    print(f"  random halves    {randoms.mean():.1%}  "
          f"(range {randoms.min():.1%} to {randoms.max():.1%} over {len(randoms)} seeds)")
    print(f"  temporal halves  {temporal:.1%}  "
          f"(calibrate before {pd.Timestamp(split_date):%Y-%m-%d}, check after)")
    if abs(randoms.mean() - TARGET_COVERAGE) > 0.015:
        print("  [WARN] random halves are not near the target. Random halves are")
        print("         exchangeable by construction, so suspect the code, not the data.")
    if randoms.min() <= temporal <= randoms.max():
        verdict = "inside"
        print("  The temporal result sits inside the random range: no sign of drift")
        print("  within the calibration period.")
    elif temporal < randoms.min():
        verdict = "below"
        print("  The temporal result is BELOW the random range: later delays ran past")
        print("  the bound more often than earlier ones. A sign of drift inside 2025;")
        print("  the 2026 test set may undercover too.")
    else:
        verdict = "above"
        print("  The temporal result is ABOVE the random range: later delays were easier")
        print("  to cover. That is still drift; the 2026 test set may overcover.")

    rule("6. WIDTH  (calibration rows)")
    lower, upper = interval(CONFORMAL_SCORE, preds, q_hat)
    # The floor: a conformal bound with no model. Score = the delay itself,
    # so everyone gets the same upper bound. Valid, but it cannot tell a
    # routine delay from a shutdown.
    floor_q, _ = conformal_quantile(y, TARGET_COVERAGE)
    print(f"  upper bound      median {np.median(upper):.1f} min, mean {upper.mean():.1f} min, "
          f"largest {upper.max():.0f} min")
    if CONFORMAL_SCORE == "two_sided":
        print(f"  lower bound      median {np.median(lower):.1f} min")
        print(f"  width            median {np.median(upper - lower):.1f} min")
    print(f"  no-model bound   {floor_q:.0f} min for every delay")
    print(f"  rows whose upper bound is below the no-model bound: "
          f"{np.mean(upper < floor_q):.0%}")

    DOCS_DIR.mkdir(exist_ok=True)
    record = {
        "score": CONFORMAL_SCORE,
        "target_coverage": TARGET_COVERAGE,
        "model": bundle["name"],
        "train_rows": int(bundle["train_rows"]),
        "n_calibration": int(n),
        "k": int(k),
        "q_hat": q_hat,
        "no_model_bound": floor_q,
        "raw_q90_coverage_calibration": raw,
        "rehearsal_random_mean": float(randoms.mean()),
        "rehearsal_random_min": float(randoms.min()),
        "rehearsal_random_max": float(randoms.max()),
        "rehearsal_temporal": temporal,
        "rehearsal_temporal_vs_random": verdict,
        "calibration_upper_median": float(np.median(upper)),
        "calibration_upper_mean": float(upper.mean()),
    }
    DAY3_CALIBRATION.write_text(json.dumps(record, indent=2))
    print(f"\nWrote {DAY3_CALIBRATION.relative_to(DOCS_DIR.parent)}")
    print("Next: write your prediction for the test set in the log, then")
    print("      python src/12_test_once.py            (dry run, shows the plan)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
