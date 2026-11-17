"""
Fix the Day 4 audit groups BEFORE anything looks at test coverage by group.

Run it with:
    python src/14_define_groups.py

Writes docs/day4_groups.json. Reads training rows (to place each delay code
in a tier) and calibration rows (only to count how many rows each group
holds). Never reads a test row. The script asserts this.

Once 16_audit.py has run, this script refuses to rewrite the groups: moving
a group boundary after seeing the audit would be slicing until something
breaks.
"""

import json
import sys

from config import CODE_TIER_CUTOFFS, DAY4_AUDIT, DAY4_GROUPS, DOCS_DIR, MIN_ROWS_PER_GROUP
from conformal import load_final_model
from data import load_population
from groups import OTHER_TIER, build_spec, code_medians


def rule(title):
    print(f"\n{title}")
    print("-" * 70)


def main():
    if DAY4_AUDIT.exists():
        print("The audit has already run on these groups (docs/day4_audit.json exists).")
        print("Changing them now would mean choosing groups after seeing the results.")
        print("Nothing written.")
        return 1

    rule("1. LOAD")
    bundle = load_final_model()
    frame = load_population()
    train = frame[frame["slice"].isin(["fit", "valid"])]
    calib = frame[frame["slice"] == "calib"]
    assert set(train["slice"]) | set(calib["slice"]) <= {"fit", "valid", "calib"}
    if bundle["train_rows"] != len(train):
        print(f"  [FAIL] model fitted on {bundle['train_rows']:,} rows, training has {len(train):,}.")
        return 1
    kept = set(bundle["encoder"].code_keep_)
    print(f"  training rows     {len(train):,}")
    print(f"  calibration rows  {len(calib):,} (counted, not scored)")
    print(f"  codes with their own model category: {len(kept)}; the rest are '{OTHER_TIER}'")

    spec = build_spec(train, calib, kept)

    rule(f"2. CODE TIERS  (training median delay, cut at {', '.join(f'{c:g}' for c in CODE_TIER_CUTOFFS)} min)")
    medians = code_medians(train, kept)
    for tier in spec["tiers"]:
        lo, hi = tier["median_range"]
        longest = medians[tier["codes"]].sort_values(ascending=False).index[:4].tolist()
        print(f"  {tier['label']:<22} {tier['n_codes']:>3} codes, {tier['training_rows']:>6,} training rows, "
              f"medians {lo:g} to {hi:g}")
        print(f"  {'':<22} longest: {', '.join(longest)}")
    other_rows = int((~train['code'].astype('string').isin(kept)).sum())
    print(f"  {OTHER_TIER:<22} {other_rows:>6,} training rows")
    if spec["merges"]:
        print(f"\n  Merged because a tier had under {MIN_ROWS_PER_GROUP} calibration rows:")
        for m in spec["merges"]:
            print(f"    {m}")
    else:
        print(f"\n  No tier needed merging (each has {MIN_ROWS_PER_GROUP}+ calibration rows).")

    rule("3. CALIBRATION ROWS PER GROUP  (counts only)")
    for grouping, counts in spec["calibration_counts"].items():
        print(f"  {grouping}")
        for group, n in counts.items():
            note = f"   description only: under the {MIN_ROWS_PER_GROUP}-row floor" if n < MIN_ROWS_PER_GROUP else ""
            print(f"    {group:<24} {n:>6,}{note}")
    tested = sum(len(c) - len(spec["descriptive_only"][g]) for g, c in spec["calibration_counts"].items())
    print(f"\n  Groups that can be flagged: {tested}. Day 4's Bonferroni correction divides by this.")

    DOCS_DIR.mkdir(exist_ok=True)
    DAY4_GROUPS.write_text(json.dumps(spec, indent=2))
    print(f"\nWrote {DAY4_GROUPS.relative_to(DOCS_DIR.parent)}")
    print("Next: python src/15_mondrian.py   (calibration only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
