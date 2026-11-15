"""
Fit four baselines on the FIT slice and score them on the VALIDATION slice.

Run it with:
    python src/08_baselines.py

Writes docs/day2_scores.csv and prints the same table.

Calibration and test rows are loaded (so the slice counts can be printed)
but never passed to any model. The script asserts this.

THE FOUR BASELINES
------------------
0. Global quantiles     One number for everyone. The floor: anything that
                        cannot beat this has learned nothing.
1. Group quantiles      Quantiles within station x hour band, falling back
                        to line x hour band, then line, when a cell is thin.
2. Linear quantile      sklearn QuantileRegressor on one-hot features.
   regression
3. Gradient boosting    HistGradientBoostingRegressor with quantile loss.

Every baseline predicts two quantiles, 0.5 and 0.9 (QUANTILES in config).

WHY MODEL log1p(delay) AND WHY THAT IS SAFE FOR QUANTILES
-----------------------------------------------------------
Delays are heavily right-skewed (skew 21.5 raw, 1.82 after log1p). Models 2
and 3 fit on log1p(min_delay) and convert back with expm1. For a mean this
would introduce bias, but quantiles pass straight through any increasing
function: the 90th percentile of log1p(Y) is log1p of the 90th percentile of
Y. So a quantile fitted on the log scale IS the quantile in minutes.

THE SCORE: PINBALL LOSS
-----------------------
For quantile q, prediction p and truth y:
    loss = q * (y - p)        if y >= p   (prediction too low)
           (1 - q) * (p - y)  if y <  p   (prediction too high)
At q = 0.9 under-predicting costs 9 times more than over-predicting, which
is what pushes a model toward the 90th percentile. At q = 0.5 it is half the
absolute error. Lower is better. Reported in minutes.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import OneHotEncoder

from config import DOCS_DIR, QUANTILES
from data import CATEGORICAL, FeatureEncoder, load_population

# The rule for choosing a baseline, fixed BEFORE seeing any scores so the
# choice cannot drift toward whichever number looks best afterwards.
SELECTION_RULE = "lowest pinball loss at q=0.9 on validation"

GROUP_LEVELS = [("station_grp", "hour_band"), ("line", "hour_band"), ("line",)]
GROUP_MIN_ROWS = 30


def pinball(y, pred, q):
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


# ---------------------------------------------------------------------------
# The baselines. Each has fit(X, y) and predict(X) -> {q: array of minutes}.
# ---------------------------------------------------------------------------
class GlobalQuantiles:
    name = "0 global quantiles"

    def fit(self, X, y):
        self.values_ = {q: float(np.quantile(y, q)) for q in QUANTILES}
        return self

    def predict(self, X):
        return {q: np.full(len(X), v) for q, v in self.values_.items()}


class GroupQuantiles:
    name = "1 group quantiles"

    def fit(self, X, y):
        frame = X.astype(str).assign(y=np.asarray(y))
        self.global_ = {q: float(np.quantile(y, q)) for q in QUANTILES}
        self.tables_ = []
        for level in GROUP_LEVELS:
            grouped = frame.groupby(list(level))["y"]
            sizes = grouped.size()
            keep = sizes[sizes >= GROUP_MIN_ROWS].index
            table = {q: grouped.quantile(q).loc[keep] for q in QUANTILES}
            self.tables_.append((level, table))
        return self

    def predict(self, X):
        frame = X.astype(str)
        out = {}
        for q in QUANTILES:
            pred = pd.Series(np.nan, index=frame.index)
            # Walk from the most specific level to the most general,
            # filling only rows that are still empty.
            for level, table in self.tables_:
                if pred.notna().all():
                    break
                keys = pd.MultiIndex.from_frame(frame[list(level)]) if len(level) > 1 \
                    else pd.Index(frame[level[0]])
                looked_up = pd.Series(table[q].reindex(keys).to_numpy(), index=frame.index)
                pred = pred.fillna(looked_up)
            out[q] = pred.fillna(self.global_[q]).to_numpy()
        return out

    def fallback_shares(self, X):
        """How many rows each level answered. Thin cells show up here."""
        frame = X.astype(str)
        answered = pd.Series(False, index=frame.index)
        shares = {}
        q = QUANTILES[0]
        for level, table in self.tables_:
            keys = pd.MultiIndex.from_frame(frame[list(level)]) if len(level) > 1 \
                else pd.Index(frame[level[0]])
            hit = pd.Series(table[q].reindex(keys).notna().to_numpy(), index=frame.index)
            new = hit & ~answered
            shares[" x ".join(level)] = new.mean()
            answered |= hit
        shares["global"] = (~answered).mean()
        return shares


class LinearQuantile:
    name = "2 linear quantile"

    def fit(self, X, y):
        self.onehot_ = OneHotEncoder(handle_unknown="ignore").fit(X)
        Z = self.onehot_.transform(X)
        target = np.log1p(y)
        # alpha=0: no penalty, plain quantile regression. "highs" is the
        # fast linear-programming solver; the default can take minutes here.
        self.models_ = {
            q: QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(Z, target)
            for q in QUANTILES
        }
        return self

    def predict(self, X):
        Z = self.onehot_.transform(X)
        return {q: np.expm1(m.predict(Z)) for q, m in self.models_.items()}


class GradientBoosting:
    name = "3 gradient boosting"

    def fit(self, X, y):
        codes = self._codes(X)
        target = np.log1p(y)
        self.models_ = {
            q: HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                categorical_features=[True] * codes.shape[1],
                max_iter=300,
                learning_rate=0.05,
                # Stop adding trees when an internal 10% hold-out of the FIT
                # rows stops improving. This hold-out is random within the
                # fit slice, which is fine: it only tunes tree count, and
                # validation is still untouched.
                early_stopping=True,
                random_state=0,
            ).fit(codes, target)
            for q in QUANTILES
        }
        return self

    @staticmethod
    def _codes(X):
        # Integer category codes. Every column is categorical to the model.
        return np.column_stack([X[c].cat.codes.to_numpy() for c in X.columns])

    def predict(self, X):
        codes = self._codes(X)
        return {q: np.expm1(m.predict(codes)) for q, m in self.models_.items()}


BASELINES = [GlobalQuantiles, GroupQuantiles, LinearQuantile, GradientBoosting]


def score(name, preds, y, seconds):
    row = {"model": name}
    for q in QUANTILES:
        row[f"pinball_q{int(q * 100)}"] = pinball(y, preds[q], q)
    q_hi = max(QUANTILES)
    # Share of validation delays at or below the predicted 90th percentile.
    # A well-calibrated q=0.9 model would give 0.90. It usually will not,
    # and that gap is exactly what conformal prediction fixes on Day 3.
    row["share_below_q90"] = float(np.mean(y <= preds[q_hi]))
    row["q90_below_q50"] = float(np.mean(preds[q_hi] < preds[min(QUANTILES)]))
    row["fit_seconds"] = round(seconds, 1)
    return row


def main():
    frame = load_population()
    fit = frame[frame["slice"] == "fit"]
    valid = frame[frame["slice"] == "valid"]

    # The guarantee depends on these never being used before Day 3.
    used_slices = set(fit["slice"]) | set(valid["slice"])
    assert used_slices <= {"fit", "valid"}, used_slices

    encoder = FeatureEncoder().fit(fit)
    X_fit, X_valid = encoder.transform(fit), encoder.transform(valid)
    y_fit = fit["min_delay"].to_numpy(dtype=float)
    y_valid = valid["min_delay"].to_numpy(dtype=float)

    print("FEATURES (category counts learned from the fit slice)")
    print("-" * 70)
    for column, n in encoder.summary().items():
        print(f"  {column:<14} {n:>4} categories")
    print(f"\nSelection rule, fixed in advance: {SELECTION_RULE}\n")

    rows = []
    for cls in BASELINES:
        model = cls()
        print(f"fitting {model.name} ...", flush=True)
        start = time.perf_counter()
        model.fit(X_fit, y_fit)
        seconds = time.perf_counter() - start
        rows.append(score(model.name, model.predict(X_valid), y_valid, seconds))
        if isinstance(model, GroupQuantiles):
            shares = model.fallback_shares(X_valid)
            print("    validation rows answered at each level: "
                  + ", ".join(f"{k} {v:.0%}" for k, v in shares.items()))

    table = pd.DataFrame(rows).set_index("model")
    print("\nVALIDATION SCORES  (minutes; lower pinball is better)")
    print("-" * 70)
    print(table.round(3).to_string())

    best = table["pinball_q90"].idxmin()
    floor = table.loc["0 global quantiles", "pinball_q90"]
    gain = 1 - table.loc[best, "pinball_q90"] / floor
    print(f"\nChosen by the rule: {best}")
    print(f"  {gain:.1%} lower q90 pinball loss than the global floor")
    print(f"  validation share below its q90: {table.loc[best, 'share_below_q90']:.1%} "
          f"(target 90%)")

    out = DOCS_DIR / "day2_scores.csv"
    table.to_csv(out)
    print(f"\nWrote {out.relative_to(DOCS_DIR.parent)}")
    print(f"Next: python src/09_fit_final.py \"{best}\"")


if __name__ == "__main__":
    main()
