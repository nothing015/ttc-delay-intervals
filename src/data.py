"""
Load the modelling population and turn it into features. Imported by every
script from Day 2 onward, so the population is defined in exactly one place.

Nothing here is run directly. Use it like this:

    from data import load_population, FeatureEncoder
    frame = load_population()
    fit = frame[frame["slice"] == "fit"]
    encoder = FeatureEncoder().fit(fit)
    X_fit = encoder.transform(fit)

WHY THE ENCODER IS FITTED ON THE FIT SLICE ONLY
-----------------------------------------------
Deciding which stations and codes are common enough to get their own
category is a choice made from data. If that choice looked at calibration or
test rows, information from those slices would leak into the model before
they are used. So the category lists are learned from the fit slice, frozen,
and applied unchanged everywhere else.
"""

import pandas as pd

from config import (
    CALIB_END,
    DELAYS_PARQUET,
    HOUR_BANDS,
    LINE_ALIASES,
    MIN_CATEGORY_ROWS,
    MIN_DELAY_THRESHOLD,
    MODEL_LINES,
    START_DATE,
    TRAIN_END,
    VALID_START,
)

SLICES = ("fit", "valid", "calib", "test")

# Categorical features every baseline sees. min_gap is deliberately absent:
# it is the gap between trains caused by the delay itself, so using it would
# be predicting the delay from its own consequence.
CATEGORICAL = ["line", "hour_band", "dayofweek", "code_grp", "station_grp"]

_BAND_OF_HOUR = {h: name for name, hours in HOUR_BANDS for h in hours}


def _check_dates():
    dates = [pd.Timestamp(d) for d in (START_DATE, VALID_START, TRAIN_END, CALIB_END)]
    if not all(a < b for a, b in zip(dates, dates[1:])):
        raise ValueError(
            "config dates must satisfy START_DATE < VALID_START < TRAIN_END "
            f"< CALIB_END, got {START_DATE}, {VALID_START}, {TRAIN_END}, {CALIB_END}"
        )


def load_population(verbose=True):
    """Read the Parquet and return the rows we model, each tagged with a slice.

    Every filter reports how many rows it removed (project rule 3).
    """
    _check_dates()
    data = pd.read_parquet(DELAYS_PARQUET)

    def report(label, frame, before):
        if verbose:
            print(f"  {label:<38} {len(frame):>8,}  -{before - len(frame):,}")

    if verbose:
        print("LOADING THE MODELLING POPULATION")
        print("-" * 70)
        print(f"  {'rows in Parquet':<38} {len(data):>8,}")

    before = len(data)
    data = data[data["min_delay"] >= MIN_DELAY_THRESHOLD].copy()
    report(f"min_delay >= {MIN_DELAY_THRESHOLD}", data, before)

    # astype("string") keeps a missing line missing (pandas 3), and isin()
    # then treats it as not-a-modelled-line, which is what we want.
    data["line"] = (
        data["line"].astype("string").str.strip().str.upper().replace(LINE_ALIASES)
    )
    before = len(data)
    data = data[data["line"].isin(MODEL_LINES)].copy()
    report(f"line in {', '.join(MODEL_LINES)}", data, before)

    data["hour_band"] = data["hour"].map(_BAND_OF_HOUR)
    if data["hour_band"].isna().any():
        raise ValueError("some hours are not covered by HOUR_BANDS in config.py")

    t = data["event_time"]
    data["slice"] = "fit"
    data.loc[(t >= pd.Timestamp(VALID_START)) & (t < pd.Timestamp(TRAIN_END)), "slice"] = "valid"
    data.loc[(t >= pd.Timestamp(TRAIN_END)) & (t < pd.Timestamp(CALIB_END)), "slice"] = "calib"
    data.loc[t >= pd.Timestamp(CALIB_END), "slice"] = "test"

    if verbose:
        counts = data["slice"].value_counts()
        print("-" * 70)
        for name in SLICES:
            n = int(counts.get(name, 0))
            print(f"  {name:<8} {n:>8,}  ({n / len(data):.1%})")
        print()
    return data.reset_index(drop=True)


class FeatureEncoder:
    """Learn category lists from the fit slice; apply them anywhere.

    Stations and codes with fewer than MIN_CATEGORY_ROWS fit rows become
    OTHER. So does any value first seen after the fit slice, which is how a
    code that appears in 2026 but never in training is handled.
    """

    def fit(self, frame):
        self.station_keep_ = self._frequent(frame["station"])
        self.code_keep_ = self._frequent(frame["code"])
        encoded = self._raw(frame)
        # Freeze the category order so every slice gets identical codes.
        self.categories_ = {
            c: sorted(encoded[c].dropna().unique().tolist()) + (
                ["OTHER"] if c in ("station_grp", "code_grp") and
                "OTHER" not in encoded[c].unique() else []
            )
            for c in CATEGORICAL
        }
        return self

    @staticmethod
    def _frequent(column):
        counts = column.astype("string").value_counts()
        return set(counts[counts >= MIN_CATEGORY_ROWS].index)

    def _raw(self, frame):
        out = pd.DataFrame(index=frame.index)
        out["line"] = frame["line"].astype("string")
        out["hour_band"] = frame["hour_band"].astype("string")
        out["dayofweek"] = frame["dayofweek"].astype(int).astype("string")
        station = frame["station"].astype("string")
        code = frame["code"].astype("string")
        out["station_grp"] = station.where(station.isin(self.station_keep_), "OTHER")
        out["code_grp"] = code.where(code.isin(self.code_keep_), "OTHER")
        return out.fillna("OTHER")

    def transform(self, frame):
        """Return a frame of pandas categoricals with the frozen categories.

        Any value outside the frozen list becomes NaN in a categorical, which
        would silently mean "missing" to a model. We map it to OTHER instead.
        """
        raw = self._raw(frame)
        out = pd.DataFrame(index=frame.index)
        for c in CATEGORICAL:
            values = raw[c].where(raw[c].isin(self.categories_[c]), "OTHER")
            out[c] = pd.Categorical(values, categories=self.categories_[c])
        return out

    def summary(self):
        return {c: len(v) for c, v in self.categories_.items()}
