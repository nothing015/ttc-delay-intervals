"""
Serve the Day 3 interval as JSON: line, station, delay code and time in,
"90% chance it clears within X minutes" out.

Two ways to run it:

    python src/18_serve.py --line YU --station "UNION STATION" --code SUDP --time "2026-03-04 08:15"
        prints one JSON answer and exits

    python src/18_serve.py --serve
        answers GET http://127.0.0.1:8000/interval?line=YU&station=UNION%20STATION&code=SUDP&time=2026-03-04T08:15
        until Ctrl+C

RETROSPECTIVE, NOT A RIDER TOOL
-------------------------------
The delay code is assigned by TTC staff, so a rider on a platform would not
have it. The endpoint shows the model can sit behind an interface; it does
not claim the interval is available at the moment a train stops.

WHAT IT DOES NOT DO
-------------------
It never reads a calibration or test row. q_hat is read from the locked
Day 3 result (docs/day3_test_results.json), not recomputed, so the interval
it returns is exactly the one Day 3 measured: from 1 minute up to the
model's predicted 90th percentile plus q_hat.

Standard library only (http.server, no FastAPI), so it needs nothing beyond
requirements.txt and the fresh-environment test covers it too. It binds to
127.0.0.1, so nothing outside this laptop can reach it.
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from config import (
    CONFORMAL_SCORE,
    DAY3_TEST_RESULTS,
    DAY4_AUDIT,
    HOUR_BANDS,
    LINE_ALIASES,
    MODEL_LINES,
    TARGET_COVERAGE,
)
from conformal import Q_HIGH, Q_LOW, interval, load_final_model, predict

# The final model was refitted on exactly this many training rows on Day 2.
# A different count means the data or the code has changed since the
# interval was measured, and q_hat would no longer belong to this model.
EXPECTED_TRAIN_ROWS = 19_804

BAND_OF_HOUR = {h: name for name, hours in HOUR_BANDS for h in hours}


class BadRequest(ValueError):
    """An input the endpoint can name and refuse, rather than guess at."""


def load():
    """Load the model and the locked Day 3 result, and check they belong together."""
    bundle = load_final_model()
    if not DAY3_TEST_RESULTS.exists():
        raise FileNotFoundError(f"{DAY3_TEST_RESULTS} not found: Day 3 has not been measured")
    day3 = json.loads(DAY3_TEST_RESULTS.read_text())

    # Each check guards the same thing: that q_hat was computed for THIS
    # model with THIS score. Mixing them silently voids the guarantee.
    problems = []
    if bundle["train_rows"] != EXPECTED_TRAIN_ROWS:
        problems.append(f"model fitted on {bundle['train_rows']:,} rows, expected {EXPECTED_TRAIN_ROWS:,}")
    if bundle["name"] != day3["model"]:
        problems.append(f"model is {bundle['name']!r}, Day 3 measured {day3['model']!r}")
    if CONFORMAL_SCORE != day3["score"] or CONFORMAL_SCORE != "upper":
        problems.append(f"config score {CONFORMAL_SCORE!r}, Day 3 used {day3['score']!r}; this endpoint serves 'upper'")
    if problems:
        raise RuntimeError("Refusing to serve: " + "; ".join(problems))

    # Group coverage is context for the answer, not part of it, so a missing
    # audit file only removes that field.
    audit = {}
    if DAY4_AUDIT.exists():
        for row in json.loads(DAY4_AUDIT.read_text())["rows"]:
            audit[(row["grouping"], row["group"])] = {
                "group": row["group"],
                "n_test_delays": row["n"],
                "coverage": round(row["coverage"], 3),
                "ci95": [round(row["ci95_low"], 3), round(row["ci95_high"], 3)],
            }
    return bundle, day3, audit


def parse(params):
    """Turn raw strings into one row the encoder understands, or refuse."""
    missing = [k for k in ("line", "station", "code", "time") if not params.get(k, "").strip()]
    if missing:
        raise BadRequest(f"missing: {', '.join(missing)}")

    # Same clean-up data.py applies to the line column, so "yus" means YU.
    line = params["line"].strip().upper()
    line = LINE_ALIASES.get(line, line)
    if line not in MODEL_LINES:
        raise BadRequest(f"line must be one of {', '.join(MODEL_LINES)}; got {params['line']!r}")

    try:
        when = pd.Timestamp(params["time"].strip())
    except (ValueError, TypeError):
        raise BadRequest(f"time must look like 2026-03-04T08:15; got {params['time']!r}") from None

    # The raw data writes stations and codes in capitals; matching that is
    # what lets "union station" find its own category instead of OTHER.
    return {
        "line": line,
        "station": params["station"].strip().upper(),
        "code": params["code"].strip().upper(),
        "time": when,
    }


def answer(bundle, day3, audit, row):
    """The interval for one delay, plus what the model saw and how far to trust it."""
    hour_band = BAND_OF_HOUR[row["time"].hour]
    frame = pd.DataFrame([{
        "line": row["line"],
        "station": row["station"],
        "code": row["code"],
        "hour_band": hour_band,
        "dayofweek": row["time"].day_of_week,  # Monday = 0, as in the Parquet
    }])

    preds = predict(bundle, frame)
    q_hat = day3["q_hat"]
    lower, upper = interval("upper", preds, q_hat)
    encoder = bundle["encoder"]
    upper_min = float(upper[0])

    return {
        "input": {
            "line": row["line"],
            "station": row["station"],
            "code": row["code"],
            "time": row["time"].isoformat(),
        },
        # What the model actually used. A station or code with fewer than 30
        # training rows is pooled into OTHER; saying so beats hiding it.
        "model_saw": {
            "hour_band": hour_band,
            "day_of_week": int(row["time"].day_of_week),
            "station": row["station"] if row["station"] in encoder.station_keep_ else "OTHER",
            "code": row["code"] if row["code"] in encoder.code_keep_ else "OTHER",
        },
        "interval_minutes": {"lower": round(float(lower[0]), 2), "upper": round(upper_min, 2)},
        "reads_as": f"{TARGET_COVERAGE:.0%} chance the delay clears within {upper_min:.1f} minutes",
        "parts": {
            "model_q90": round(float(preds[Q_HIGH][0]), 3),
            "q_hat": round(q_hat, 4),
            "model_median_no_guarantee": round(float(preds[Q_LOW][0]), 2),
        },
        "guarantee": (
            f"marginal: {TARGET_COVERAGE:.0%} averaged over all delays, not within any one group. "
            f"Measured once at {day3['coverage']:.1%} on {day3['n_test']:,} test delays "
            f"({day3['test_from']} to {day3['test_to']})."
        ),
        "test_coverage_in_this_delays_groups": {
            grouping: audit[(grouping, group)]
            for grouping, group in (("line", row["line"]), ("hour_band", hour_band))
            if (grouping, group) in audit
        },
        "caveat": "retrospective: needs the TTC delay code, which is assigned after the delay starts",
    }


def serve(bundle, day3, audit, port):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body):
            data = json.dumps(body, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                return self._send(200, {
                    "usage": "/interval?line=YU&station=UNION%20STATION&code=SUDP&time=2026-03-04T08:15",
                    "lines": list(MODEL_LINES),
                })
            if url.path != "/interval":
                return self._send(404, {"error": f"no route {url.path}; try /interval"})
            # parse_qs gives lists; one value per field is all we accept.
            params = {k: v[-1] for k, v in parse_qs(url.query).items()}
            try:
                return self._send(200, answer(bundle, day3, audit, parse(params)))
            except BadRequest as err:
                return self._send(400, {"error": str(err)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving on http://127.0.0.1:{port}/interval  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="90% upper bound on a logged TTC subway delay, as JSON.")
    parser.add_argument("--line")
    parser.add_argument("--station")
    parser.add_argument("--code")
    parser.add_argument("--time", help='e.g. "2026-03-04 08:15"')
    parser.add_argument("--serve", action="store_true", help="run an HTTP server instead of answering once")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    bundle, day3, audit = load()
    if args.serve:
        serve(bundle, day3, audit, args.port)
        return 0
    params = {k: getattr(args, k) or "" for k in ("line", "station", "code", "time")}
    try:
        print(json.dumps(answer(bundle, day3, audit, parse(params)), indent=2))
    except BadRequest as err:
        print(json.dumps({"error": str(err)}, indent=2))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
