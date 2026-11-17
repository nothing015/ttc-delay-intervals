"""
Which audit group each delay belongs to. Imported by 14_define_groups.py,
15_mondrian.py and 16_audit.py. Nothing here is run directly.

THE THREE GROUPINGS (fixed on Day 1; config.AUDIT_GROUPINGS)
-------------------------------------------------------------
line        YU, BD, SHP
hour_band   the six bands in config.HOUR_BANDS
code_tier   delay codes grouped by how long they ran IN TRAINING. Each code
            the model knows (30+ training rows) gets its median delay over
            the 19,804 training rows, and the medians are cut at
            config.CODE_TIER_CUTOFFS. Codes the model pools into OTHER (rare
            in training, or first seen later) form one more group.

Day 1 left "delay-code category" undefined. One group per code is too thin
(most codes have far fewer than 100 calibration rows), and the code's
prefix (MU, SU, PU...) would bury MUPR1, a 72-minute code, among routine
ones. Grouping by typical length tests the Day 1 prediction directly: that
long-delay codes undercover.

WHY THE TIERS ARE LEARNED FROM TRAINING ROWS ONLY
-------------------------------------------------
Which tier a code is in must not depend on how well it was covered, or the
audit would find what it went looking for. Training rows decide each code's
tier. Calibration rows are used only to COUNT how many rows each tier holds,
so a tier with fewer than MIN_ROWS_PER_GROUP can be merged with its
neighbour. Counting rows says nothing about coverage. Test rows are not
used at all.

HOW A THIN TIER IS MERGED
-------------------------
While some tier has fewer than MIN_ROWS_PER_GROUP calibration rows, the
smallest one is merged with whichever neighbour holds fewer rows (an end
tier has only one neighbour). Tiers stay contiguous ranges of median delay.
The OTHER group is never merged: it has no neighbour on that scale. If it
is thin it is reported for description only and left out of the flags.

Labels read as ranges of the training median: "median 5-10 min" means more
than 5 and at most 10.
"""

import numpy as np
import pandas as pd

from config import CODE_TIER_CUTOFFS, HOUR_BANDS, MIN_ROWS_PER_GROUP, MODEL_LINES

OTHER_TIER = "rare or new code"


def tier_label(lo, hi):
    if lo is None and hi is None:
        return "all known codes"
    if lo is None:
        return f"median <= {hi:g} min"
    if hi is None:
        return f"median > {lo:g} min"
    return f"median {lo:g}-{hi:g} min"


def code_medians(train, kept_codes):
    """Median training delay of every code the model has its own category for."""
    frame = pd.DataFrame({
        "code": train["code"].astype("string").to_numpy(),
        "y": train["min_delay"].to_numpy(dtype=float),
    })
    frame = frame[frame["code"].isin(kept_codes)]
    return frame.groupby("code")["y"].median().sort_index()


def build_spec(train, calib, kept_codes):
    """Learn the code tiers (training), merge thin ones (calibration counts).

    Returns a plain dict, saved as docs/day4_groups.json and read back by
    every later script, so all of them use exactly the same groups.
    """
    cutoffs = [float(c) for c in CODE_TIER_CUTOFFS]
    medians = code_medians(train, kept_codes)
    # searchsorted(side="left") puts a median equal to a cutoff in the lower
    # tier, which is what "at most 5" means.
    index = np.searchsorted(cutoffs, medians.to_numpy(), side="left")
    edges = [None, *cutoffs, None]
    tiers = []
    for i in range(len(cutoffs) + 1):
        codes = sorted(medians.index[index == i].tolist())
        if codes:  # an empty tier has nothing to audit
            tiers.append({"lo": edges[i], "hi": edges[i + 1], "codes": codes})

    calib_codes = calib["code"].astype("string")

    def count(tier):
        return int(calib_codes.isin(tier["codes"]).sum())

    merges = []
    while len(tiers) > 1:
        counts = [count(t) for t in tiers]
        i = int(np.argmin(counts))
        if counts[i] >= MIN_ROWS_PER_GROUP:
            break
        if i == 0:
            j = 1
        elif i == len(tiers) - 1:
            j = i - 1
        else:
            j = i - 1 if counts[i - 1] <= counts[i + 1] else i + 1
        a, b = sorted((i, j))
        merges.append(
            f"{tier_label(tiers[a]['lo'], tiers[a]['hi'])} ({counts[a]} rows) + "
            f"{tier_label(tiers[b]['lo'], tiers[b]['hi'])} ({counts[b]} rows)"
        )
        tiers[a:b + 1] = [{"lo": tiers[a]["lo"], "hi": tiers[b]["hi"],
                           "codes": sorted(tiers[a]["codes"] + tiers[b]["codes"])}]

    for tier in tiers:
        tier["label"] = tier_label(tier["lo"], tier["hi"])
        tier["n_codes"] = len(tier["codes"])
        tier["training_rows"] = int(train["code"].astype("string").isin(tier["codes"]).sum())
        tier["median_range"] = [float(medians[tier["codes"]].min()),
                                float(medians[tier["codes"]].max())]

    spec = {
        "cutoffs": cutoffs,
        "min_rows_per_group": MIN_ROWS_PER_GROUP,
        "training_rows": int(len(train)),
        "kept_codes": int(len(kept_codes)),
        "tiers": tiers,
        "other_label": OTHER_TIER,
        "code_to_tier": {code: t["label"] for t in tiers for code in t["codes"]},
        "merges": merges,
        "order": {
            "line": list(MODEL_LINES),
            "hour_band": [name for name, _ in HOUR_BANDS],
            "code_tier": [t["label"] for t in tiers] + [OTHER_TIER],
        },
    }
    groups = add_groups(calib, spec)
    spec["calibration_counts"] = {
        g: {k: int(v) for k, v in groups[g].value_counts().reindex(spec["order"][g], fill_value=0).items()}
        for g in spec["order"]
    }
    spec["descriptive_only"] = {
        g: [k for k, n in counts.items() if n < MIN_ROWS_PER_GROUP]
        for g, counts in spec["calibration_counts"].items()
    }
    return spec


def add_groups(frame, spec):
    """One string column per grouping, for the rows of frame."""
    out = pd.DataFrame(index=frame.index)
    out["line"] = frame["line"].astype("string")
    out["hour_band"] = frame["hour_band"].astype("string")
    code = frame["code"].astype("string")
    tier = code.map(spec["code_to_tier"])
    out["code_tier"] = tier.where(tier.notna(), spec["other_label"]).astype("string")
    for g in ("line", "hour_band", "code_tier"):
        unknown = set(out[g].dropna().unique()) - set(spec["order"][g])
        if out[g].isna().any() or unknown:
            raise ValueError(f"{g} has values outside the fixed groups: {sorted(unknown)[:5]}")
    return out


def tested_groups(spec, grouping):
    """Groups with enough calibration rows to be flagged; the rest are description only."""
    thin = set(spec["descriptive_only"][grouping])
    return [g for g in spec["order"][grouping] if g not in thin]
