# Day 3 journal: conformal intervals, one look at the test set

**Project:** prediction intervals for TTC subway delay length, with a coverage guarantee
**Date:** 16 November 2026
**Status:** done. Results committed and pushed (`4c36da1`).

This is my record of Day 3: what I built, what I found, what I decided and
why. As with Days 1 and 2, each section is written so I can answer questions
about it without opening the code.

---

## 1. Day 3 in one minute

| | |
|---|---|
| Goal | Turn Day 2's saved model into 90% intervals with split conformal prediction, calibrate on the calibration slice, measure the test set once. |
| Output | `docs/day3_calibration.json`, `docs/day3_test_results.json`, `docs/figures/day3_coverage.png`, and `data/processed/day3_test_rows.parquet` for Day 4 (not in git) |
| Score | `"upper"`: interval from 1 minute up to the predicted q90 plus `q_hat` |
| Correction | `q_hat` = 0.0835 minutes (n = 6,563, k = 5,908) |
| Headline number | **92.1% test coverage** (95% CI 91.4% to 92.7%) against a 90% target |
| Honest caveat | Above the 89.0% to 91.0% expected range: 2025 and 2026 were not exchangeable |
| Biggest surprise | Conformal barely had to do anything; the model's own q90 was nearly calibrated |
| Not yet checked | Coverage inside any line, hour band or delay code. That is Day 4. |

The one sentence for Day 3: **the guarantee held, but 2026 delays came in
shorter than 2025 predicted, so the intervals were wider than they needed
to be.**

---

## 2. Concepts I need to be able to explain

### Conformity score

A number per row saying how wrong the model was. Bigger means worse. Any
score gives a valid guarantee; the score only decides what the interval
looks like.

### Why not absolute residuals in minutes

The plan said "plain split conformal on absolute residuals". That gives the
prediction plus or minus one fixed number of minutes. My delays are skewed
(median 5, p90 14, maximum 716), so a symmetric interval is the wrong shape:
its lower end falls below 1 minute, which no delay in the population can be,
and a five-minute delay gets the same width as an hour-long shutdown.

### The score I chose: `"upper"`

```
score    = actual delay − predicted 90th percentile        (minutes)
interval = 1 minute  up to  predicted q90 + q_hat
covered  ⇔  score ≤ q_hat
```

It reads as "90% chance it clears within X minutes", which is the project's
question. It builds on the q90 model, where Day 2 found the model helps most
(21.2% better than the floor, against 6.8% for the median). This is the
one-sided version of conformalized quantile regression: start from a model
that already predicts a quantile, and use held-out rows only to correct how
far it misses. The lower end at 1 minute costs nothing, because every delay
in the population is at least 1.

### Exchangeable, in plain words

Ten equally good runners: before the race, each is equally likely to finish
in any place. Exchangeable data is the same idea. The new delay is not
special compared with the calibration delays, so shuffling them together
changes nothing. For me that means assuming a 2026 delay looks like a 2025
delay. It holds only approximately, and Day 3 found it did not hold.

### Why `n + 1`

Line up the n calibration scores and the one new score, smallest to largest.
That is n + 1 positions. If the rows are exchangeable, the new score is
equally likely to land in any of them. It is covered when it lands in the
bottom k, so the chance is **k / (n + 1)**. Choosing
`k = ceil((n + 1) × 0.9)` makes that at least 0.9.

Using n instead undercovers: with n = 10, `ceil(0.9 × 10)` = 9 gives
9/11 = 82%, while the n + 1 rule gives k = 10 and 10/11 = 91%.

For my data: k = ceil(6,564 × 0.9) = 5,908, so the chance is
5,908 / 6,564 = 0.9001.

**The argument never uses the model.** Only the ranking of scores and
exchangeability. That is why a bad model still gives valid intervals, just
wider ones.

### When k is bigger than n

Five scores: k = ceil(6 × 0.9) = 6, and there is no 6th smallest. Using the
largest score covers only 5/6 = 83%. The only way to promise 90% is to also
cover the top position, where the new score beats every calibration score,
which needs no upper limit. So `q_hat` is infinite. The code returns
`math.inf` for this. It cannot happen with 6,563 rows.

### "At least 90%" does not mean "more is fine"

My scores are continuous (an integer delay minus a fractional prediction),
so ties are essentially impossible. Under exchangeability the coverage is
then almost exactly k / (n + 1) = 90.0%, not "90% or more". A result well
above 90% is not extra safety the theorem gave me. It is evidence that the
assumption failed, in the harmless direction.

### Expected range vs confidence interval

Two different numbers printed side by side:

- **95% CI (91.4% to 92.7%)**: noise from counting coverage on a finite
  test set. "How sure am I of the 92.1%?"
- **Expected range (89.0% to 91.0%)**: where coverage should land 95 times
  in 100 *if* calibration and test are exchangeable. It adds the noise from
  estimating `q_hat` on a finite calibration set:
  `0.9 ± 1.96 × √(0.9 × 0.1 × (1/6,563 + 1/7,448))`.

A result outside the expected range points to drift, not to a bug. Mine was
outside it, and so was the whole CI.

### The rehearsal

Run on calibration rows only, so it can be repeated and changes nothing.

- **Random halves**: calibrate on a random half, check the other. Random
  halves are exchangeable by construction, so this must give about 90%. If
  not, the code is wrong.
- **Temporal halves**: calibrate on the earlier half, check the later half.
  If this falls outside the random range, delays drifted inside 2025, and
  2026 may drift too.

### The no-model bound

Conformal with score = the delay itself: everyone gets the same upper bound
(14 minutes). Just as valid as my intervals, but it cannot tell a routine
delay from a shutdown. The model earns its place only if most delays get a
tighter bound than that. 65% did.

### Marginal, not conditional

92.1% is averaged over all 7,448 test delays. One group could sit at 75%,
balanced by others at 95%. Overall overcoverage makes this *easier* to hide,
not harder. Day 4 looks.

### Why the test set is measured once

Looking, changing something and looking again tunes the method to the test
set, and the number then describes my choices instead of the method.
`12_test_once.py` writes `docs/day3_test_results.json` last, as a lock: once
it exists the script prints the stored result and refuses to measure again.

---

## 3. Setup changes

Added to the end of `src/config.py`:

| Setting | Value | Why |
|---|---|---|
| `CONFORMAL_SCORE` | `"upper"` | Fixed before any code ran, like Day 2's selection rule |
| `DAY3_CALIBRATION` | `docs/day3_calibration.json` | `q_hat` and the rehearsal; committed |
| `DAY3_TEST_RESULTS` | `docs/day3_test_results.json` | The one test result, and the lock; committed |
| `DAY3_TEST_ROWS` | `data/processed/day3_test_rows.parquet` | One row per test delay for Day 4; ignored by git (`data/`) |

`docs/figures/` is ignored by `.gitignore`. I force-added the one Day 3
figure (`git add -f`) because it is the headline result and a reader on
GitHub will not run my scripts.

---

## 4. The pipeline, Day 3 additions

| Script | What it does |
|---|---|
| `conformal.py` | The method: scores, `conformal_quantile`, interval, `is_covered`, model loading, and a self-test on made-up numbers. Not run directly. |
| `11_calibrate.py` | Self-test, then `q_hat` from the calibration slice, the rehearsal and the width comparison. Never reads a test delay. Re-runnable. |
| `12_test_once.py` | Dry run by default. With `--confirm`, measures test coverage once, writes the results and the per-row file, then refuses to run again. Recomputes `q_hat` and stops if it differs from the calibration record. |
| `13_plot.py` | Draws the coverage figure from the saved results. Never touches the test set. |

Safety checks the scripts make: the model must have been fitted on exactly
19,804 rows; calibration must start at or after `TRAIN_END`; the score in
`config.py` must match the one used for calibration.

---

## 5. What went wrong, and what each one taught me

**1. I could not start the hand cases.**
I did not know what "exchangeable" meant, so the `n + 1` argument had
nothing to stand on. The runners analogy fixed it, then a worked case with
three scores at 50%.
*Lesson: when a proof will not start, the stuck word is usually a
definition. Find a small concrete version of it.*

**2. I wrote "9 out of 9" for the first hand case.**
k = 9 and `q_hat` = 9 were right, but I forgot that the new score takes a
position in the line too. There are 10 positions, so the chance is 9/10.
9 out of 9 would be 100%, a guarantee too good to be true.
*Lesson: this slip is exactly the `+ 1`.*

**3. I said the argument never uses "assumptions".**
It uses one: exchangeability. What it never uses is the model.
*Lesson: that sentence is the interview answer, so get the noun right.*

**4. Three commits went in missing files.**
The journals were not saved before `git add`; the Step 5 entry and test
prediction were not in `log.md` the first time. Each time the commit
message described more than the commit contained.
*Lesson: check `git show --stat` after every commit. A commit message is
not evidence of what is in it.*

**5. The Day 3 prediction had vanished from `log.md`.**
The "Prediction for Day 3" block from the end of Day 2 was not in the
committed file. I wrote it again and committed it (`02fa047`, 21:48), before
`11_calibrate.py` ran, and dated it as written on 16 November rather than
pretending it was older.
*Lesson: a prediction is pre-registered when it is committed, not when I
intend to write it.*

**6. The code and the score setting were uncommitted until just before
`--confirm`.**
`git status` showed `config.py` modified and the four scripts untracked.
Committed in `b5c835d` before the measurement.
*Lesson: commit the code that will measure before it measures, so the repo
shows what produced the number.*

**7. A multi-line paste broke in PowerShell.**
Two commands ran together and a third was reordered. Same problem as Day 1.
*Lesson: one command per line, and paste each one separately.*

**8. My Day 1 journal had `k` slightly wrong.**
It used 6,565 calibration rows (before the line filter) and got 5,910. The
modelling population has 6,563, which gives 5,908.

**9. The dry run ran before my test prediction was written.**
Harmless, because the dry run measures nothing, but the runbook put the
prediction first. The prediction's heading says "before the test
measurement", which is true.

---

## 6. Results

### Calibration (`11_calibrate.py`, 6,563 rows, 2025-04-01 to 2025-12-31)

| | |
|---|---|
| Model's own q90 coverage, before conformal | 89.8% (validation gave 88.5%) |
| n, k, level | 6,563, 5,908, 0.9002 |
| `q_hat` | **0.0835 minutes**: every predicted q90 raised by under a tenth of a minute |
| Coverage on calibration rows themselves | 90.0% (true by construction, not a result) |

### Rehearsal (calibration rows only)

| | Coverage |
|---|---|
| Random halves, 20 seeds | 90.2% (range 88.9% to 90.8%) |
| Temporal halves (split at 2025-08-19) | 91.8%, **above** the random range |

Random halves near 90%: the code behaves. Temporal halves above: later 2025
delays were slightly easier to cover than earlier ones. One split, so a
hint, not a measurement. It turned out to point the right way.

### Width (calibration rows)

Upper bound median 11.6 minutes, mean 13.2, largest 143, against 14 minutes
for the no-model bound. 65% of delays got a tighter bound than 14.

### Test (`12_test_once.py --confirm`, once, 7,448 rows, 2026-01-01 to 2026-09-30)

| | |
|---|---|
| **Coverage** | **92.1%**, 95% CI 91.4% to 92.7% |
| Expected range if exchangeable | 89.0% to 91.0% |
| Model's own q90, before conformal | 91.8% |
| No-model bound (14 min for everyone) | 92.9% |
| Upper bound | median 11.5 min, mean 13.1 min |
| Tighter than the no-model bound | 65% of delays |

How to read it:

- **The guarantee held.** Coverage is above 90%.
- **The result is outside the expected range, and so is the whole CI.**
  Calibration and test were not exchangeable.
- **The same rule, same model, gave 89.8% on 2025 and 91.8% on 2026.** The
  no-model bound of 14 minutes covered at least 90.0% of calibration delays
  and 92.9% of test delays. So delays themselves ran shorter in 2026, not
  only the model's errors. I can describe the shift; nothing here says why.
- **Conformal added 0.3 points** (91.8% to 92.1%), because `q_hat` was tiny.
- **The model still earns its place**: 65% of delays got a tighter bound
  than one number for everyone, the same share as in calibration.

### The coverage figure

`docs/figures/day3_coverage.png`: coverage measured against coverage asked
for, at ten levels from 50% to 95%, all measured in the single pass.

- All ten points sit above the diagonal; no 95% interval crosses zero.
- The gap is about 3 to 4 points from 55% to 85%, 2.1 at 90% and 1.4 at 95%.
- So the shift is across the whole distribution, not only the upper tail.
- The ten points come from the same 7,448 delays, so they are one finding,
  not ten.

### My predictions against the results

| Predicted | Got | |
|---|---|---|
| q90 coverage on calibration about 88.5% (87.5% to 89.5%) | 89.8% | just outside my range |
| `q_hat` about 1 minute, under 2 | 0.0835 minutes | right sign, about 12 times too big |
| No-model bound about 14 minutes | 14 | right |
| Test coverage about 90.5% (89% to 92%), leaning above 90% | 92.1% | right direction, 0.1 points above my range |

I anchored on validation's 88.5%, which came from 2,523 rows (about ±0.6
points of noise). The rehearsal's 91.8% was the better guide to the test
set, and I under-weighted it.

---

## 7. Decisions, each with its reason

| Decision | Reason |
|---|---|
| Score `"upper"`, not absolute residuals | Skewed delays need an asymmetric interval; the question is an upper bound |
| Score fixed in `config.py` before any code ran | So the choice cannot drift toward the best-looking result |
| Lower end of the interval at 1 minute | Every delay in the population is at least 1 minute |
| `q_hat` from all 6,563 calibration rows, whatever the rehearsal said | The rehearsal only tells me what to expect |
| Predictions committed before each measurement | The commit time proves the order |
| Code and config committed before `--confirm` | The repo records what produced the number |
| Test measured once; script refuses a second run | Rule 2, written as code |
| Nothing changed after seeing 92.1% | Recalibrating on 2026 would tune to the test set |
| Adaptive conformal inference named, not applied | It is the method built for drift; a next step for the write-up |
| Figure force-added despite `docs/figures/` being ignored | It is the headline result, and GitHub readers will not rebuild it |
| No looking at groups today | Day 4's groups were fixed on Day 1 |

---

## 8. My prediction for Day 4 (written before opening the per-row file)

Already on record:

- Day 1: undercoverage, if any, will be in rare long-delay code groups such
  as `MUPR1`; hour of day will be a weak grouping; if nothing undercovers
  once coverage has confidence intervals, I report a negative result.
- Day 2 journal: the `OTHER` code group will undercover first, then rare
  long-delay codes.

> Written 17 November, after the Step 6 calibration rehearsal and before
> `16_audit.py` read the per-row file.
>
> Overall coverage of 92.1% does change what I expect: the 2026 shift
> lifted coverage by about 2 points everywhere, so groups that were 1 to
> 2 points low on calibration should land near 90%. I expect **none of
> the 12 flaggable groups** to undercover under the Bonferroni rule. The
> lowest will be the "median > 5 min" code tier (about 90%), then the
> early and overnight hour bands. "Rare or new code" will sit above
> overall (about 93%), against my Day 2 prediction. The long-code tier
> merged into "> 5 min" (43 calibration rows), so the Day 1 `MUPR1`
> prediction cannot be tested.

---

## 9. Questions I should be able to answer quickly

**Why does conformal prediction work without the model being correct?**
The guarantee comes from ranking the scores, not from the model. The new
score is equally likely to land anywhere among the n + 1, so the chance it
is in the bottom k is k / (n + 1). A worse model gives bigger scores, so a
bigger `q_hat` and wider intervals, but the same coverage.

**Why `n + 1` and not `n`?**
The new score takes a position in the line too. With n = 10, using n gives
9/11 = 82%; using n + 1 gives 10/11 = 91%.

**What is the one assumption?**
Exchangeability: calibration and test rows could be shuffled together
without anyone noticing. Mine are 2025 and 2026, and it did not fully hold.

**Your coverage was 92.1%. Isn't more than 90% good?**
The intervals are valid, but wider than needed, and the number is outside
the range exchangeability predicts. With continuous scores, exchangeable
data would give almost exactly 90%. So 92.1% is evidence of drift, in the
harmless direction.

**How do you know it is drift and not a bug?**
The self-test passed, the random-half rehearsal gave 90.2%, and the
no-model bound, which involves no model at all, also covered more in 2026
(92.9%) than in 2025. The temporal rehearsal had already flagged a shift
within 2025.

**What did conformal add, if the model's q90 was nearly right?**
A guarantee. The model's own coverage was 90.6% on its training rows, 88.5%
on validation, 89.8% on calibration and 91.8% on test, with nothing to say
where it would land next. Conformal measures the miss on held-out rows.

**Why did `q_hat` come out at only 0.08 minutes?**
The model's q90 already covered 89.8% of calibration delays, so the 5,908th
smallest score was just above zero.

**Why a score built on the 90th percentile instead of absolute error?**
Delays are skewed, so a symmetric interval is the wrong shape. Starting from
a quantile model lets the bound be long for a shutdown and short for a
routine delay.

**What would you do with drift like this in a real system?**
Not recalibrate on the test set. I would use adaptive conformal inference,
which updates the correction as new data arrives.

**Does 92.1% hold for every line or delay code?**
No. It is marginal. Overall overcoverage can hide a group that undercovers.
Day 4 checks.

**Why commit the prediction before running anything?**
The commit time proves the prediction came before the number. On Day 2 I
logged the selection rule after the run; this time the order is in the
git history.

---

## 10. Commands I run to check everything

```powershell
python src/07_verify.py           # expect "All 31 checks passed"
python -c "import sys; sys.path.insert(0,'src'); from conformal import self_test; [print('PASS' if ok else 'FAIL', name) for name, ok, detail in self_test()]"
                                  # expect four PASS lines
python src/11_calibrate.py        # q_hat 0.0835; safe to re-run
python src/12_test_once.py        # prints the stored result and refuses to measure again
python src/13_plot.py             # redraws the figure from saved results
```

---

## 11. Open items

- [ ] Write the AI acknowledgement below, and the Day 3 paragraph in
      `log.md`
- [ ] Fill in section 8 and commit it before Day 4 opens the per-row file
- [ ] In `log.md` Step 7, "the same bound covered 90% of 2025 calibration
      delays" should say "at least 90.0%"
- [ ] Day 4: decide whether to write the coverage audit myself and have it
      reviewed, or have Claude write it
- [ ] Day 5: add the coverage figure and the drift finding to the README

---

## AI acknowledgement

Claude wrote the conformal code (`conformal.py`, `11_calibrate.py`, `12_test_once.py`, `13_plot.py`) and drafted this journal and my predictions from my output; before running any of it I learned what exchangeability means and worked the n + 1 hand cases on paper, mistakes included (section 5), so I can defend each step without the code.
