"""
The conformal step, in one place. Imported by 11_calibrate.py and
12_test_once.py. Nothing here is run directly.

THE WHOLE METHOD IN FOUR LINES
------------------------------
1. For every calibration row, compute a SCORE: how wrong the model was.
2. Sort the scores and take the k-th smallest, with k = ceil((n + 1) * 0.90).
   Call it q_hat.
3. For a new row, the interval is every delay whose score would be <= q_hat.
4. Theorem: that interval contains the truth at least 90% of the time,
   whatever the model is, if calibration and new rows are exchangeable.

WHY (n + 1) AND NOT n
---------------------
Put the n calibration scores and the one new score in a line, smallest to
largest. If the rows are exchangeable, the new score is equally likely to
land in any of the n + 1 positions. It is covered when it lands in one of
the first k positions, so the chance is k / (n + 1). Choosing
k = ceil((n + 1) * 0.90) makes that chance at least 0.90. Using n instead
would undercover slightly; with 6,563 rows the difference is tiny, but the
proof needs the + 1.

THE TWO SCORES THIS PROJECT SUPPORTS (config.CONFORMAL_SCORE)
-------------------------------------------------------------
"upper"      score = actual - predicted 90th percentile, in minutes.
             Positive when the delay ran past the model's upper bound.
             Interval: from 1 minute up to (predicted q90 + q_hat).
             Reads as "90% chance it clears within X minutes".

"two_sided"  score = |log1p(actual) - log1p(predicted median)|.
             Interval: the median stretched by the same factor both ways,
             so it is asymmetric in minutes, as skewed delays should be.

Both give the same guarantee. They differ in what the interval looks like.
"""

import importlib
import math

import joblib
import numpy as np

from config import MIN_DELAY_THRESHOLD, MODELS_DIR, QUANTILES

SCORES = ("upper", "two_sided")

Q_LOW, Q_HIGH = min(QUANTILES), max(QUANTILES)


def conformal_quantile(scores, coverage):
    """Return (q_hat, k): the k-th smallest score, k = ceil((n + 1) * coverage).

    If k is larger than n there are too few calibration rows to promise the
    coverage at all, and the only honest answer is an infinite interval.
    """
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("no calibration scores")
    if np.isnan(scores).any():
        raise ValueError("a calibration score is NaN; a prediction or a delay is missing")
    # round() guards against float noise such as 9.000000000000002, which
    # ceil would turn into 10 and silently widen every interval.
    k = math.ceil(round((n + 1) * coverage, 9))
    if k > n:
        return math.inf, k
    return float(scores[k - 1]), k  # k-th smallest; Python counts from 0


def conformity_scores(kind, y, preds):
    """How wrong the model was on each row. Bigger means worse."""
    y = np.asarray(y, dtype=float)
    if kind == "upper":
        return y - preds[Q_HIGH]
    if kind == "two_sided":
        return np.abs(np.log1p(y) - np.log1p(preds[Q_LOW]))
    raise ValueError(f"CONFORMAL_SCORE must be one of {SCORES}, got {kind!r}")


def interval(kind, preds, q_hat):
    """Turn predictions plus q_hat into (lower, upper), both in minutes.

    The interval is exactly the set of delays whose score is <= q_hat, cut
    off below at 1 minute. That cut costs nothing: every delay in the
    population is at least 1 minute, so nothing below it can be the truth.
    """
    if kind == "upper":
        upper = preds[Q_HIGH] + q_hat
        lower = np.full(len(upper), float(MIN_DELAY_THRESHOLD))
        return lower, upper
    if kind == "two_sided":
        centre = np.log1p(preds[Q_LOW])
        lower = np.maximum(np.expm1(centre - q_hat), float(MIN_DELAY_THRESHOLD))
        upper = np.expm1(centre + q_hat)
        return lower, upper
    raise ValueError(f"CONFORMAL_SCORE must be one of {SCORES}, got {kind!r}")


def is_covered(kind, y, preds, q_hat):
    """True where the truth is inside the interval.

    Written as "score <= q_hat" on purpose: it is the same comparison used
    to pick q_hat, so calibration and test are judged by one rule.
    """
    return conformity_scores(kind, y, preds) <= q_hat


def load_final_model():
    """Load Day 2's saved model, encoder and metadata."""
    # The saved object refers to classes defined in 08_baselines.py, so that
    # module has to be importable before joblib can rebuild them.
    importlib.import_module("08_baselines")
    path = MODELS_DIR / "final.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python src/09_fit_final.py \"3 gradient boosting\""
        )
    return joblib.load(path)


def predict(bundle, frame):
    """Predicted quantiles in minutes for the rows of frame: {0.5: ..., 0.9: ...}."""
    return bundle["model"].predict(bundle["encoder"].transform(frame))


def self_test(coverage=0.90):
    """Check the implementation where the right answer is known.

    Returns a list of (name, passed, detail). Uses made-up numbers only, so
    it cannot leak anything about calibration or test.
    """
    results = []

    # Three cases small enough to do by hand.
    q, k = conformal_quantile([5, 1, 9, 3, 7, 2, 8, 4, 6], coverage)
    results.append(("9 scores 1..9: k = ceil(10 * 0.9) = 9, q_hat = 9",
                    (q, k) == (9.0, 9), f"got q_hat {q}, k {k}"))
    q, k = conformal_quantile(list(range(1, 20)), coverage)
    results.append(("19 scores 1..19: k = ceil(20 * 0.9) = 18, q_hat = 18",
                    (q, k) == (18.0, 18), f"got q_hat {q}, k {k}"))
    q, k = conformal_quantile([1, 2, 3, 4, 5], coverage)
    results.append(("5 scores: k = ceil(6 * 0.9) = 6 > 5, so q_hat is infinite",
                    math.isinf(q) and k == 6, f"got q_hat {q}, k {k}"))

    # Simulation: 20,000 worlds, each with 100 calibration scores and one new
    # score drawn the same way (so they are exchangeable by construction).
    # Theory says the new score is covered with chance k / (n + 1) = 91 / 101.
    rng = np.random.default_rng(0)
    n, trials = 100, 20_000
    draws = rng.exponential(size=(trials, n + 1))  # skewed, like delays
    k = math.ceil(round((n + 1) * coverage, 9))
    q_hats = np.sort(draws[:, :n], axis=1)[:, k - 1]
    simulated = float(np.mean(draws[:, n] <= q_hats))
    theory = k / (n + 1)
    results.append((f"simulated coverage matches theory {theory:.3f} within 0.010",
                    abs(simulated - theory) < 0.010, f"simulated {simulated:.3f}"))
    return results
