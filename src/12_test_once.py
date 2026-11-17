"""
Measure coverage on the TEST slice. Once.

Run it with:
    python src/12_test_once.py             dry run: shows the plan, measures nothing
    python src/12_test_once.py --confirm   the real measurement

Writes docs/day3_test_results.json, and data/processed/day3_test_rows.parquet
with one row per test delay for Day 4's audit.

WHY THIS SCRIPT REFUSES TO RUN TWICE
------------------------------------
Project rule 2: the test set is measured once, at the end. If you look at
test coverage, change something and look again, you are tuning to the test
set and the number stops meaning what it claims. So once the results file
exists, this script prints the stored result and stops. Day 4 does not
re-measure: it slices the saved per-row file.

Everything reported here was decided before the run: the score, the target
coverage, the no-model comparison and the list of levels for the plot.

WHAT IT REPORTS
---------------
coverage          share of test delays inside their interval, with a 95%
                  confidence interval for test-set sampling noise
expected range    where coverage should land if calibration and test are
                  exchangeable. Wider than the confidence interval, because
                  q_hat is itself estimated from a finite calibration set
before conformal  the model's own q90 on the same rows, for comparison
no-model bound    one upper bound for everyone, calibrated the same way.
                  Also valid; shows what the model buys in width
coverage curve    the same recipe at levels from 50% to 95%, for the plot
"""

import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    CALIB_END,
    CONFORMAL_SCORE,
    DAY3_CALIBRATION,
    DAY3_TEST_RESULTS,
    DAY3_TEST_ROWS,
    DOCS_DIR,
    TARGET_COVERAGE,
)
from conformal import (
    Q_HIGH,
    Q_LOW,
    conformal_quantile,
    conformity_scores,
    interval,
    load_final_model,
    predict,
)
from data import load_population

# Fixed in advance, for the coverage plot.
CURVE_LEVELS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def wilson(successes, n, z=1.96):
    """95% confidence interval for a proportion. Better behaved than
    p +/- 1.96 * se when p is near 0 or 1."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - spread), float(centre + spread)


def expected_range(target, n_calibration, n_test, z=1.96):
    """Where test coverage should land 95% of the time IF calibration and
    test rows are exchangeable.

    Two sources of noise add up: q_hat comes from a finite calibration set
    (variance about t(1 - t) / n_calibration), and coverage is then counted
    on a finite test set (t(1 - t) / n_test). A result outside this range
    is evidence against exchangeability, not against the code.
    """
    se = np.sqrt(target * (1 - target) * (1 / n_calibration + 1 / n_test))
    return float(target - z * se), float(target + z * se)


def show(result):
    print(f"  score               {result['score']}")
    print(f"  test rows           {result['n_test']:,}  ({result['test_from']} to {result['test_to']})")
    print(f"  q_hat               {result['q_hat']:.4f}  (from {result['n_calibration']:,} calibration rows)")
    print()
    print(f"  COVERAGE            {result['coverage']:.1%}   "
          f"95% CI {result['coverage_ci_low']:.1%} to {result['coverage_ci_high']:.1%}   "
          f"target {result['target_coverage']:.0%}")
    print(f"  expected range      {result['expected_low']:.1%} to {result['expected_high']:.1%}   "
          f"(if calibration and test are exchangeable)")
    print(f"  before conformal    {result['raw_q90_coverage_test']:.1%}   (model's own q90, uncorrected)")
    print(f"  no-model bound      {result['no_model_coverage_test']:.1%}   "
          f"({result['no_model_bound']:.0f} min for every delay)")
    print()
    print(f"  upper bound         median {result['upper_median']:.1f} min, mean {result['upper_mean']:.1f} min")
    if result["score"] == "two_sided":
        print(f"  width               median {result['width_median']:.1f} min")
    print(f"  rows with a tighter bound than the no-model bound: {result['share_tighter_than_no_model']:.0%}")


def main():
    confirm = "--confirm" in sys.argv[1:]

    if DAY3_TEST_RESULTS.exists():
        result = json.loads(DAY3_TEST_RESULTS.read_text())
        print("THE TEST SET HAS ALREADY BEEN MEASURED")
        print("-" * 70)
        print(f"  measured at         {result['measured_at']}")
        show(result)
        print("\nThis script will not measure it again (project rule 2). Day 4 reads")
        print(f"{DAY3_TEST_ROWS.name} instead of re-running anything.")
        return 1

    if not DAY3_CALIBRATION.exists():
        print("No calibration record. Run: python src/11_calibrate.py")
        return 1
    calibration = json.loads(DAY3_CALIBRATION.read_text())
    if calibration["score"] != CONFORMAL_SCORE:
        print(f"config.CONFORMAL_SCORE is {CONFORMAL_SCORE!r} but calibration used "
              f"{calibration['score']!r}. Re-run 11_calibrate.py.")
        return 1

    bundle = load_final_model()
    frame = load_population(verbose=False)
    calib = frame[frame["slice"] == "calib"]
    test = frame[frame["slice"] == "test"]
    assert test["event_time"].min() >= pd.Timestamp(CALIB_END), "test overlaps calibration"

    # Recompute q_hat rather than trusting the file, and insist they agree.
    # A mismatch means the model, config or data changed since calibration.
    calib_scores = conformity_scores(
        CONFORMAL_SCORE, calib["min_delay"].to_numpy(dtype=float), predict(bundle, calib)
    )
    q_hat, k = conformal_quantile(calib_scores, TARGET_COVERAGE)
    if not np.isclose(q_hat, calibration["q_hat"]) or len(calib_scores) != calibration["n_calibration"]:
        print(f"q_hat is now {q_hat:.4f} but 11_calibrate.py recorded "
              f"{calibration['q_hat']:.4f}. Something changed. Re-run 11_calibrate.py.")
        return 1

    if not confirm:
        print("DRY RUN: nothing has been measured")
        print("-" * 70)
        print(f"  model               {bundle['name']}")
        print(f"  score               {CONFORMAL_SCORE}")
        print(f"  q_hat               {q_hat:.4f}  (k = {k:,} of {len(calib_scores):,} calibration rows)")
        print(f"  target coverage     {TARGET_COVERAGE:.0%}")
        print(f"  test rows waiting   {len(test):,}")
        low, high = expected_range(TARGET_COVERAGE, len(calib_scores), len(test))
        print(f"  expected range      {low:.1%} to {high:.1%}  (if calibration and test are exchangeable)")
        print("\nThis can be run exactly once. Before you do:")
        print("  1. your prediction for test coverage is written in docs/log.md")
        print("  2. you are not planning to change anything after seeing the number")
        print("\nThen: python src/12_test_once.py --confirm")
        return 0

    # ---- the one measurement -------------------------------------------
    y = test["min_delay"].to_numpy(dtype=float)
    preds = predict(bundle, test)
    scores = conformity_scores(CONFORMAL_SCORE, y, preds)
    covered = scores <= q_hat
    lower, upper = interval(CONFORMAL_SCORE, preds, q_hat)
    n = len(y)
    ci_low, ci_high = wilson(int(covered.sum()), n)
    expected_low, expected_high = expected_range(TARGET_COVERAGE, len(calib_scores), n)

    no_model_bound = float(calibration["no_model_bound"])
    curve = []
    for level in CURVE_LEVELS:
        q_level, _ = conformal_quantile(calib_scores, level)
        hits = int(np.sum(scores <= q_level))
        lo, hi = wilson(hits, n)
        curve.append({"level": level, "coverage": hits / n, "ci_low": lo, "ci_high": hi})

    result = {
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "score": CONFORMAL_SCORE,
        "model": bundle["name"],
        "target_coverage": TARGET_COVERAGE,
        "n_calibration": int(len(calib_scores)),
        "q_hat": q_hat,
        "n_test": int(n),
        "test_from": f"{test['event_time'].min():%Y-%m-%d}",
        "test_to": f"{test['event_time'].max():%Y-%m-%d}",
        "coverage": float(covered.mean()),
        "coverage_ci_low": ci_low,
        "coverage_ci_high": ci_high,
        "expected_low": expected_low,
        "expected_high": expected_high,
        "raw_q90_coverage_test": float(np.mean(y <= preds[Q_HIGH])),
        "no_model_bound": no_model_bound,
        "no_model_coverage_test": float(np.mean(y <= no_model_bound)),
        "upper_median": float(np.median(upper)),
        "upper_mean": float(upper.mean()),
        "width_median": float(np.median(upper - lower)),
        "share_tighter_than_no_model": float(np.mean(upper < no_model_bound)),
        "curve": curve,
    }

    # One row per test delay, so Day 4 can slice coverage by group without
    # predicting on the test set again.
    groups = bundle["encoder"].transform(test)
    rows = pd.DataFrame({
        "event_time": test["event_time"].to_numpy(),
        "line": test["line"].astype("string").to_numpy(),
        "hour_band": test["hour_band"].astype("string").to_numpy(),
        "dayofweek": test["dayofweek"].to_numpy(),
        "station": test["station"].astype("string").to_numpy(),
        "code": test["code"].astype("string").to_numpy(),
        "station_grp": groups["station_grp"].astype("string").to_numpy(),
        "code_grp": groups["code_grp"].astype("string").to_numpy(),
        "min_delay": y,
        "pred_q50": preds[Q_LOW],
        "pred_q90": preds[Q_HIGH],
        "lower": lower,
        "upper": upper,
        "score": scores,
        "covered": covered,
    })

    # Write the per-row file first and the results file last. The results
    # file is the lock, so if anything fails before it, nothing has been
    # shown and the run can be repeated.
    DAY3_TEST_ROWS.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(DAY3_TEST_ROWS, index=False)
    DOCS_DIR.mkdir(exist_ok=True)
    DAY3_TEST_RESULTS.write_text(json.dumps(result, indent=2))

    print("TEST-SET RESULT  (measured once)")
    print("-" * 70)
    show(result)
    print()
    coverage = result["coverage"]
    if expected_low <= coverage <= expected_high:
        print("  Coverage is inside the expected range: consistent with the guarantee.")
        print("  This is MARGINAL coverage, averaged over all test delays. It says")
        print("  nothing yet about any one line, hour band or code. That is Day 4.")
    elif coverage < expected_low:
        print("  Coverage is BELOW the expected range. The code passed its self-test,")
        print("  so the likely cause is drift: 2026 delays do not look like the 2025")
        print("  calibration delays (exchangeability fails). Report it as it is.")
        print("  Do not adjust anything and re-measure.")
    else:
        print("  Coverage is ABOVE the expected range. Intervals are wider than 2026")
        print("  needed: also drift, in the harmless direction. Report it as it is.")
        print("  Do not adjust anything and re-measure.")
    print(f"\nWrote {DAY3_TEST_RESULTS.relative_to(DOCS_DIR.parent)}")
    print(f"Wrote {DAY3_TEST_ROWS.relative_to(DOCS_DIR.parent)}  ({len(rows):,} rows, for Day 4)")
    print("Next: python src/13_plot.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
