"""
Refit the chosen baseline on ALL training rows (fit + valid) and save it for
Day 3.

Run it with the model name 08_baselines.py printed, in quotes:
    python src/09_fit_final.py "3 gradient boosting"

Writes models/final.joblib. Day 3 loads it, predicts on the calibration
slice, and computes the conformal quantile from those predictions.

WHY REFIT
---------
The validation slice was held back only to CHOOSE a model. Once the choice is
made, those three months are just more training data, and the most recent
three at that. Leaving them out would make the model slightly worse for no
gain. Calibration and test are still not touched: the encoder and the model
see training rows only.
"""

import importlib
import sys

import joblib
import numpy as np

from config import MODELS_DIR
from data import FeatureEncoder, load_population

baselines = importlib.import_module("08_baselines")
BY_NAME = {cls.name: cls for cls in baselines.BASELINES}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in BY_NAME:
        print("Usage: python src/09_fit_final.py \"<model name>\"")
        print("Model names:", ", ".join(f'"{n}"' for n in BY_NAME))
        return 1

    name = sys.argv[1]
    frame = load_population()
    train = frame[frame["slice"].isin(["fit", "valid"])]
    assert set(train["slice"]) <= {"fit", "valid"}

    encoder = FeatureEncoder().fit(train)
    model = BY_NAME[name]().fit(encoder.transform(train),
                                train["min_delay"].to_numpy(dtype=float))

    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / "final.joblib"
    joblib.dump({"name": name, "encoder": encoder, "model": model,
                 "train_rows": len(train)}, path)

    # A sanity check that is safe to look at: predictions on TRAINING rows.
    # Nothing here measures coverage on calibration or test.
    preds = model.predict(encoder.transform(train))
    y = train["min_delay"].to_numpy(dtype=float)
    print(f"Refitted {name} on {len(train):,} training rows")
    print(f"  in-sample share below q90: {np.mean(y <= preds[0.9]):.1%} "
          "(in-sample, so optimistic by design)")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
