# How long until a TTC subway delay clears? Prediction intervals with a checked coverage guarantee

Nuaiman Hasan · November 2026 · [github.com/nothing015/ttc-delay-intervals](https://github.com/nothing015/ttc-delay-intervals)

## Summary

I built 90% prediction intervals for the length of a logged TTC subway delay, using split conformal prediction on top of a quantile gradient-boosting model. On 7,448 delays from January to September 2026, measured once, the intervals covered **92.1%** of delays (95% CI 91.4% to 92.7%) against a 90% target. **Marginal coverage holds; no fixed group is shown to undercover at this power:** none of 12 groups fixed in advance (3 lines, 6 hour bands, 3 delay-code tiers) fails a Bonferroni-corrected test. But 92.1% sits above the 89.0% to 91.0% that exchangeable data would produce. Calibration (2025) and test (2026) were not exchangeable: 2026 delays ran shorter, so the intervals were wider than needed.

## 1. The question

A rider told "expect 5 minutes" cannot decide whether to wait or walk. In my data the median delay is 5 minutes but the 90th percentile is 14, so a rider given the typical figure faces 14 minutes or more one time in ten. The useful statement is an upper bound with a stated reliability, "90% chance service resumes within X minutes", plus evidence that the 90% is real on data the model never saw.

The target is `min_delay`, the minutes of delay an incident caused, **given that one has been logged**; I do not predict whether a delay happens. The model uses the TTC's delay code, which staff assign after the fact, so this is a retrospective analysis, not a tool a rider could use on a platform.

## 2. Data

The source is the TTC Subway Delay Data on the City of Toronto Open Data Portal (Open Government Licence – Toronto), January 2023 to September 2026: 97,136 logged incidents. I started in 2023 to keep one operating regime; mixing in the pandemic years would weaken the exchangeability the guarantee rests on. Only 34,163 incidents (35.2%) caused a delay of at least one minute. Keeping Lines 1, 2 and 4 drops 348 more (339 from Line 3, closed July 2023): **33,815 delays**, with median 5 minutes, 90th percentile 14, maximum 716. The split is by date, never at random:

| Slice | Dates | Delays | Used for |
|---|---|---|---|
| Fit | 2023-01 to 2024-12 | 17,281 | training the baselines |
| Validation | 2025-01 to 2025-03 | 2,523 | choosing one model |
| Calibration | 2025-04 to 2025-12 | 6,563 | the conformal correction only |
| Test | 2026-01 to 2026-09 | 7,448 | measured once |

One trap is worth naming: a portal file named `.xlsx` was really CSV text, and the first build silently lost every row from 2025 on. The build now checks contents, not names, and every filter reports the rows it removes.

## 3. Method

**The model.** I scored four baselines on validation by pinball loss at the 90th percentile, a rule fixed in code before any score was printed. Gradient boosting with quantile loss on `log1p(min_delay)` won: 1.774 minutes against 2.251 for one global quantile, 21.2% lower. Linear quantile regression was within 1.3%, a tie. A percentile per station and hour band was *worse* than the global number (2.316): a tail quantile from 30 rows is noise. Features are line, hour band, day of week, delay code and station; the winner was refitted on all 19,804 training rows.

**The conformal step.** A quantile model has no coverage guarantee on new data: boosting covered 88.5% of validation delays. For each calibration delay I compute a score, *actual delay minus predicted 90th percentile*. With n = 6,563 scores, q̂ is the k-th smallest, k = ⌈(n + 1) × 0.9⌉ = 5,908. A new delay's interval runs from 1 minute up to its predicted 90th percentile plus q̂. This one-sided form of conformalized quantile regression was fixed before any code ran; symmetric intervals on absolute residuals would give a 5-minute delay and an hour-long shutdown the same width.

The guarantee comes from ranks, not from the model. If calibration and test delays are exchangeable, a new score is equally likely to land in any of the n + 1 positions among the calibration scores, so it falls in the bottom k with probability k / (n + 1) ≥ 0.9. A worse model gives a larger q̂ and wider intervals, but the same coverage.

## 4. Marginal coverage

The model's own 90th percentile covered 89.8% of calibration delays, so the correction was tiny: **q̂ = 0.0835 minutes**. A rehearsal on calibration rows gave 90.2% on random halves (88.9% to 90.8% over 20 seeds) but 91.8% when the later half was checked against the earlier: a first hint of drift.

On the test set, measured once, coverage was **92.1%** (95% CI 91.4% to 92.7%). The guarantee held. The median upper bound was 11.5 minutes, and 65% of delays got a tighter bound than the 14 minutes a no-model bound gives everyone.

![Figure 1](figures/day3_coverage.png)

*Figure 1. Test coverage against coverage asked for, at ten levels. Every level overcovers.*

But 92.1% is not extra safety. Exchangeable data with continuous scores gives close to k / (n + 1) = 90.0%, and 89.0% to 91.0% allowing for noise in both sets; the whole confidence interval lies above that. It is drift, not a bug: random halves recovered 90%, and the no-model 14-minute bound covered at least 90.0% of 2025 delays but 92.9% of 2026 delays. Delays got shorter across the whole distribution (Figure 1). That direction is harmless for validity; the intervals were wider than 2026 needed.

## 5. Coverage inside groups

The guarantee is **marginal**, an average. If SHP (304 test delays, 4.1%) fell to 80%, the rest would only need about 92.6% for the total to still read 92.1%. So I audited groups, under a design committed before the audit ran:

- **Groups:** 3 lines, 6 hour bands, and delay codes in tiers by training median (cuts at 5, 10, 20 minutes) plus "rare or new code". Not station: only 21 of 277 stations reach 100 calibration rows.
- **Intervals:** a bootstrap over whole days (2,000 draws), since delays on one bad day are not independent.
- **Rule:** a group undercovers only if its Bonferroni-adjusted interval (z = 2.87 for 12 groups) lies wholly below 90%. Uncorrected, 12 tests at 95% give a false flag about 46% of the time.

| Grouping | Group | n | Coverage | 95% CI | Adjusted CI |
|---|---|---|---|---|---|
| Line | YU | 3,977 | 93.1% | 92.2–93.9 | 91.8–94.3 |
| | BD | 3,167 | 90.7% | 89.5–91.9 | 88.9–92.5 |
| | SHP | 304 | 93.1% | 90.3–95.9 | 89.0–97.2 |
| Hour band | overnight 00–04 | 505 | 89.9% | 87.0–92.8 | 85.6–94.2 |
| | early 05–06 | 646 | 91.3% | 88.7–94.0 | 87.4–95.2 |
| | AM peak 07–09 | 1,194 | 93.3% | 91.8–94.8 | 91.1–95.5 |
| | midday 10–15 | 2,087 | 93.1% | 91.9–94.3 | 91.3–94.9 |
| | PM peak 16–18 | 1,347 | 91.0% | 89.3–92.7 | 88.5–93.5 |
| | evening 19–23 | 1,669 | 91.7% | 90.2–93.1 | 89.5–93.8 |
| Code tier | median ≤ 5 min | 5,373 | 92.8% | 92.0–93.5 | 91.7–93.9 |
| | median > 5 min | 1,528 | 89.6% | 87.9–91.3 | 87.0–92.1 |
| | rare or new code | 547 | 92.0% | 88.5–95.4 | 86.9–97.0 |

*Table 1. Test coverage by group; intervals resample whole days.*

![Figure 2](figures/day4_group_coverage.png)

*Figure 2. Table 1 drawn. Thin bars are the adjusted intervals the rule uses; every one crosses 90%.*

**No group undercovers, and none would be flagged even on its unadjusted interval.** This is the negative result I committed on Day 1 to report rather than slicing until something broke. Two patterns, neither a finding at this power: the two lowest groups, the "median > 5 min" tier (89.6%) and overnight (89.9%), were also the lowest on calibration; and BD (90.7%) sits 2.4 points below YU (93.1%) though the two were level on calibration.

**Power is the honest limit.** Adjusted intervals run about ±1 to ±3 points in large groups and ±4 to ±5 in small ones, so the audit rules out large undercoverage, not a shortfall of a point or two. **One prediction became untestable.** I expected the longest codes (`MUPR1`, median 72.5 minutes) to undercover, but the 10 to 20 minute tier was empty and the tier above 20 had only 43 calibration rows. The pre-set rule merged it into its neighbour, where long codes are under 3% of rows. I kept the rule and recorded this before the audit ran.

## 6. Would group-wise (Mondrian) conformal help?

Mondrian conformal runs the same procedure inside each group, giving each its own q̂ and its own guarantee. I tested it on calibration halves only; choosing it after seeing the test groups, then scoring it on test, would tune the method to the test set.

On random halves, exchangeable by construction, it lifted the three groups the global q̂ left low to about 90% ("median > 5 min" 87.6% → 89.9%, overnight 88.4% → 90.3%, early 87.8% → 90.1%), for 1.9 to 3.7 extra minutes of median bound. On temporal halves, with the 2025 drift, it made things **worse**: the spread across groups widened in every grouping (lines 4.5 → 11.2 points), and SHP fell to 81.4% because its q̂ came from about 145 earlier rows. Mondrian fixes imbalance between groups, not drift over time.

## 7. Limitations

- **Marginal only.** The audit covers 12 fixed groups, not single codes, stations, or combinations.
- **Exchangeability failed**, harmlessly this time; drift the other way would break coverage silently.
- **Retrospective.** Without the delay code the guarantee would hold, but intervals would likely be wider.
- **The long tail is untested.** The codes most likely to fail are too rare to audit.

## 8. Next steps

The main gap is drift. **Adaptive conformal inference** (Gibbs and Candès, 2021) updates the correction as delays arrive, raising it after misses and lowering it after hits, and targets long-run coverage under distribution shift. With a hundred times the data the conformal step would stay trivial, but most stations and codes would clear 100 calibration rows, so Mondrian could run at the level riders care about. A clean test of either needs months published after September 2026.

## References

- Angelopoulos, A. N. and Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511.
- Romano, Y., Patterson, E. and Candès, E. (2019). *Conformalized Quantile Regression.* NeurIPS.
- Gibbs, I. and Candès, E. (2021). *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS.
- Vovk, V., Gammerman, A. and Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer. (Mondrian conformal prediction.)
- City of Toronto. *TTC Subway Delay Data.* Open Data Portal, Open Government Licence – Toronto.
