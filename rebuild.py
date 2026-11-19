"""
Rebuild everything from nothing, in one command, from the repo root:

    python rebuild.py

Downloads the raw files from Toronto Open Data, builds the Parquet, runs the
Day 1 acceptance test, refits the final model, and asks it for one interval.

It does NOT re-measure anything. The Day 3 test result and the Day 4 audit
were measured once and are committed in docs/ (day3_test_results.json,
day4_audit.json); their scripts print the stored result and refuse to run
again. Rebuilding the model is what lets you check those results belong to
it: 09_fit_final.py should report 19,804 training rows.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("download the raw files", ["src/01_download.py"]),
    ("build the Parquet", ["src/02_build.py"]),
    ("Day 1 acceptance test", ["src/07_verify.py"]),
    ("refit the final model", ["src/09_fit_final.py", "3 gradient boosting"]),
    ("one example interval", ["src/18_serve.py", "--line", "YU", "--station", "UNION STATION",
                              "--code", "SUDP", "--time", "2026-03-04 08:15"]),
]


def main():
    for i, (label, args) in enumerate(STEPS, 1):
        print(f"\n=== {i}/{len(STEPS)}  {label}: python {' '.join(args)}", flush=True)
        # sys.executable is the Python running this file, so the steps use
        # the same environment (the activated .venv) rather than whatever
        # "python" happens to mean on PATH.
        result = subprocess.run([sys.executable, *args], cwd=ROOT)
        if result.returncode != 0:
            print(f"\nStopped at step {i} ({label}): exit code {result.returncode}.")
            return result.returncode
    print("\nRebuilt. The measured results are in docs/; nothing was re-measured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
