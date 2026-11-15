"""
Optional: how much does a random split flatter the model?

Run it with:
    python src/10_leakage.py

Uses TRAINING rows only (fit + valid), so calibration and test stay untouched.
Fits the same gradient-boosting median model two ways and scores each on
held-out rows of the same size:

  temporal   train on fit, score on valid (the last three months)
  random     shuffle fit + valid, hold out the same number of rows at random

Repeated over several random seeds, because one random split is itself noise.

WHAT TO EXPECT, HONESTLY
------------------------
A random split helps a model when the future leaks into the past: trends over
time, or near-copies of the same incident landing on both sides. These
features carry no date (no month, no year), so the gap may be small. A small
gap is a finding too: it says the temporal split costs little here. The
temporal split is still the correct protocol, because only it matches how
the model would be used.
"""

import importlib

import numpy as np

from data import FeatureEncoder, load_population

baselines = importlib.import_module("08_baselines")
SEEDS = range(5)


def median_loss(train, held_out):
    encoder = FeatureEncoder().fit(train)
    model = baselines.GradientBoosting().fit(
        encoder.transform(train), train["min_delay"].to_numpy(dtype=float)
    )
    preds = model.predict(encoder.transform(held_out))
    return baselines.pinball(held_out["min_delay"].to_numpy(dtype=float), preds[0.5], 0.5)


def main():
    frame = load_population()
    train = frame[frame["slice"].isin(["fit", "valid"])].reset_index(drop=True)
    fit, valid = train[train["slice"] == "fit"], train[train["slice"] == "valid"]

    temporal = median_loss(fit, valid)

    randoms = []
    for seed in SEEDS:
        held = train.sample(n=len(valid), random_state=seed)
        randoms.append(median_loss(train.drop(held.index), held))
    randoms = np.array(randoms)

    print("PINBALL LOSS AT q=0.5 ON HELD-OUT TRAINING ROWS  (lower looks better)")
    print("-" * 70)
    print(f"  temporal split   {temporal:.3f}")
    print(f"  random split     {randoms.mean():.3f}  "
          f"(range {randoms.min():.3f} to {randoms.max():.3f} over {len(randoms)} seeds)")
    gap = randoms.mean() / temporal - 1
    direction = "lower" if gap < 0 else "higher"
    print(f"\n  the random split's average loss is {abs(gap):.1%} {direction} than the temporal one")
    if randoms.max() < temporal:
        print("  Every random split looks better than the honest one: that is the flattery.")
    elif randoms.min() > temporal:
        print("  The random splits look WORSE. The last three months are easier than")
        print("  average, which is a drift finding rather than a leakage one.")
    else:
        print("  The temporal result sits inside the random range: little leakage")
        print("  with these features. Report that, and keep the temporal split anyway.")


if __name__ == "__main__":
    main()
