"""
Export what the static page needs to answer "how long until it clears?" in
a browser, with no server and no Python.

    python src/19_export_page.py

Writes docs/site/bounds.bin and docs/site/meta.json, which docs/index.html
reads. GitHub Pages serves them from the docs/ folder.

WHY A LOOKUP TABLE WORKS HERE
-----------------------------
Every feature the model sees is categorical: line, hour band, day of week,
code group, station group. So the model can only ever give a finite number
of different answers, and this script asks it for all of them, once. The
page then looks the answer up instead of running the model. It is the same
model, the same q_hat and the same interval as src/18_serve.py; the script
checks that on the README example before writing anything.

WHAT IT DOES NOT READ
---------------------
No calibration or test row. The station lists per line come from training
rows only, and the group coverage shown on the page is the locked Day 4
audit, read from docs/day4_audit.json. Nothing new is measured, and the page
offers no way to slice the test set beyond the 12 groups fixed in advance.
"""

import importlib
import itertools
import json
import sys

import numpy as np
import pandas as pd

from config import CODE_KEY_FILE, DAY4_AUDIT, DOCS_DIR, HOUR_BANDS, RAW_DIR
from conformal import Q_HIGH, predict
from data import load_population

serve = importlib.import_module("18_serve")

OUT_DIR = DOCS_DIR / "site"
# Bounds are stored as whole hundredths of a minute in an unsigned 16-bit
# integer: exact to the endpoint's two decimals, and half the size of floats.
SCALE = 100
MAX_STORABLE = 65_535 / SCALE


def main():
    bundle, day3, audit = serve.load()  # same checks as the endpoint: 19,804 rows, score, model
    encoder = bundle["encoder"]
    cats = encoder.categories_

    # The grid, in the encoder's own frozen category order. C order: the
    # last axis (station) varies fastest; the page computes the same index.
    axes = ["line", "hour_band", "dayofweek", "code_grp", "station_grp"]
    sizes = [len(cats[a]) for a in axes]
    rows = list(itertools.product(*(cats[a] for a in axes)))
    frame = pd.DataFrame(rows, columns=["line", "hour_band", "dayofweek", "code", "station"])
    frame["dayofweek"] = frame["dayofweek"].astype(int)
    print(f"Grid: {' x '.join(map(str, sizes))} = {len(frame):,} combinations")

    preds = predict(bundle, frame)
    upper = np.asarray(preds[Q_HIGH], dtype=float) + day3["q_hat"]
    if upper.max() >= MAX_STORABLE:
        sys.exit(f"Largest bound {upper.max():.1f} min does not fit the 16-bit format")
    stored = np.round(upper * SCALE).astype("<u2")  # little-endian, as the browser reads it

    # The check that matters: the table must give the endpoint's answer.
    example = serve.parse({"line": "YU", "station": "UNION STATION", "code": "SUDP",
                           "time": "2026-03-04 08:15"})
    expected = serve.answer(bundle, day3, audit, example)["interval_minutes"]["upper"]
    model_saw = serve.answer(bundle, day3, audit, example)["model_saw"]
    key = (example["line"], model_saw["hour_band"], str(model_saw["day_of_week"]),
           model_saw["code"], model_saw["station"])
    index = np.ravel_multi_index([cats[a].index(k) for a, k in zip(axes, key)], sizes)
    looked_up = stored[index] / SCALE
    if abs(looked_up - expected) > 0.006:
        sys.exit(f"Lookup gives {looked_up}, endpoint gives {expected}: not writing anything")
    print(f"Check: README example, table {looked_up:.2f} min, endpoint {expected:.2f} min")

    # Which stations to offer under each line: those seen on that line in
    # TRAINING rows. Without this, the page would offer Kennedy on Line 4.
    pop = load_population(verbose=False)
    train = pop[pop["slice"].isin(["fit", "valid"])]
    station_grp = encoder.transform(train)["station_grp"].astype(str)
    stations_by_line = {
        line: sorted(set(station_grp[train["line"] == line]) - {"OTHER"})
        for line in cats["line"]
    }

    # Code descriptions from the TTC's own key, where it has one; the key is
    # known to be incomplete (5 codes missing), so a code may have none.
    key_path = RAW_DIR / CODE_KEY_FILE
    descriptions = {}
    if key_path.exists():
        key_df = pd.read_csv(key_path)
        descriptions = dict(zip(key_df["CODE"].astype(str).str.strip(),
                                key_df["DESCRIPTION"].astype(str).str.strip()))

    audit_rows = json.loads(DAY4_AUDIT.read_text())["rows"]
    meta = {
        "axes": axes,
        "categories": {a: cats[a] for a in axes},
        "scale": SCALE,
        "hour_bands": {name: [min(h), max(h)] for name, h in HOUR_BANDS},
        "stations_by_line": stations_by_line,
        "code_descriptions": {c: descriptions.get(c, "") for c in cats["code_grp"]},
        "q_hat": day3["q_hat"],
        "test": {k: day3[k] for k in ("coverage", "coverage_ci_low", "coverage_ci_high",
                                      "n_test", "test_from", "test_to", "expected_low",
                                      "expected_high")},
        "audit": [{k: r[k] for k in ("grouping", "group", "n", "coverage", "ci95_low",
                                     "ci95_high", "ci_adj_low", "ci_adj_high", "flag")}
                  for r in audit_rows],
        "check": {"input": "YU, UNION STATION, SUDP, 2026-03-04 08:15", "upper": expected},
    }

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "bounds.bin").write_bytes(stored.tobytes())
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"Wrote {OUT_DIR / 'bounds.bin'} ({stored.nbytes:,} bytes) and meta.json")
    print(f"Bounds range from {upper.min():.1f} to {upper.max():.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
