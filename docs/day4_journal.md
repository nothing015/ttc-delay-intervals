# Day 4 journal: coverage inside each group, one look

**Project:** prediction intervals for TTC subway delay length, with a coverage guarantee
**Date:** 17 November 2026
**Status:** done. Audit run once, committed and pushed (`dd92fba`).

This is my record of Day 4: what I built, what I found, what I decided and
why. As with Days 1 to 3, each section is written so I can answer questions
about it without opening the code.

---

## 1. Day 4 in one minute

| | |
|---|---|
| Goal | Test coverage inside each line, hour band and delay-code tier, with intervals and a verdict under a rule fixed before the run. Mondrian checked on calibration only. |
| Output | `docs/day4_groups.json`, `docs/day4_mondrian.json`, `docs/day4_audit.json` (the lock), `docs/day4_audit.csv`, `docs/figures/day4_group_coverage.png` |
| Groups | 12 flaggable: 3 lines, 6 hour bands, 3 code tiers |
| Rule | A group undercovers if its Bonferroni-adjusted interval (family-wise 95%, day bootstrap) lies entirely below 90% |
| Headline | **No group undercovers. None is even "possible".** A negative result. |
| Lowest groups | "median > 5 min" code tier 89.6% (95% CI 87.9 to 91.3), overnight 89.9% (87.0 to 92.8) |
| Honest caveat | The audit rules out large undercoverage, not a shortfall of a point or two |
| Biggest surprise | No known code has a training median between 10 and 20 minutes, so the long-code tier merged away and Day 1's `MUPR1` prediction became untestable |
| Biggest lesson | Mondrian fixes imbalance between groups. It does not fix drift over time. |

The one sentence for Day 4: **the 90% held inside every group I fixed in
advance, but the two groups that were low on calibration are low again on
test, and this data cannot tell me whether that is real.**

---

## 2. Concepts I need to be able to explain

### Marginal is not conditional

92.1% is a weighted average over all 7,448 test delays. A small group can
collapse without moving it. If SHP (304 delays, 4.1%) fell to 80%, the rest
would only need about 92.6% for the overall figure to still read 92.1%, a
shift smaller than the overall figure's own confidence interval. The only
way to see a group is to look inside it.

### Mondrian (group-balanced) conformal

The Day 3 recipe run separately inside each group: that group's
calibration scores, its own k = ceil((n_group + 1) × 0.9), its own
`q_hat`. The n + 1 argument then holds inside every group, if rows are
exchangeable within the group.

- A group the global bound undercovers gets a **bigger** `q_hat`: a higher
  bound, a wider interval.
- A group it overcovers gets a **smaller** `q_hat`: width given back.
- Cost: a small group's `q_hat` comes from few rows, so it is noisy. Hence
  about 100 calibration rows per group.
- Mondrian does not choose the groups. I fixed them before looking.

### Mondrian fixes groups, not time

On random halves of calibration (exchangeable by construction) Mondrian
lifted every low group to about 90%. On temporal halves (with the 2025
drift) it made the spread across groups **wider** in every grouping, and
SHP fell to 81.4%: its own `q_hat` came from about 145 earlier rows and the
later rows had moved. A per-group correction learned from old data is still
old. The method built for drift is adaptive conformal inference (Gibbs and
Candès, 2021).

### Code tiers

Each code the model knows (30+ training rows, 64 codes) gets its median
delay over the 19,804 training rows; the medians are cut at 5, 10 and 20
minutes. Codes the model pools into `OTHER` form "rare or new code".
Learned from training rows only, so a tier cannot depend on how well it was
covered.

### Why tiers merge

A tier under 100 calibration rows merges with its smaller neighbour, so
tiers stay contiguous ranges of median delay. Counting calibration rows
says nothing about coverage, so this does not peek. "Rare or new code" has
no neighbour on that scale and is never merged.

### The day bootstrap and the design effect

One bad day produces many long delays together, and if the bound misses
one it tends to miss the others. Rows from the same day are not
independent. So the intervals resample **whole days** of the test period
(2,000 times, seed 0), keeping every delay on each drawn day.

Design effect = (bootstrap SE / independent-rows SE)². 1 means days added
no extra noise; 2 means the group carries the information of half as many
independent rows. Mine run 1.0 to 1.5, except rare or new code at 2.3.

### Bonferroni

With 12 groups each tested at 95% and none truly undercovering, the chance
that at least one is flagged anyway is 1 − 0.95¹² ≈ 46%. Testing each at
0.05 / 12 (z = 2.87 instead of 1.96) keeps the chance of any false flag at
or below 5%. The price is power: a real shortfall of a point or two can go
unflagged.

### "possible"

The ordinary 95% interval is below 90% but the adjusted one is not. Worth
naming, not a finding. On Day 4 no group was even "possible".

### Read against two lines

90% is what the guarantee promises. 92.1% is what 2026 delivered overall.
A group at 90.7% (BD) does not undercover the guarantee, but it sits 2.4
points below YU and below the overall figure: it gained less from the
2026 shift.

### Why Mondrian is not applied to the test set

Choosing Mondrian after seeing which test groups miss, then scoring it on
test, would tune the method to the test set. A clean test needs delays the
project has never seen, such as months published after September 2026.

### Why the audit runs once

The audit does not re-measure the test set: it slices Day 3's saved
per-row result. It still runs once, because if I could redraw groups and
re-run, I could slice until something broke. `docs/day4_audit.json` is the
lock; `14_define_groups.py` also refuses to rewrite the groups once it
exists.

---

## 3. Setup changes

Added to the end of `src/config.py`:

| Setting | Value | Why |
|---|---|---|
| `AUDIT_GROUPINGS` | `("line", "hour_band", "code_tier")` | Fixed on Day 1 |
| `CODE_TIER_CUTOFFS` | `(5, 10, 20)` | Overall median 5, p90 14, long codes past 20 |
| `AUDIT_BOOTSTRAP_REPS` | `2000` | Enough for a stable standard error |
| `AUDIT_SEED` | `0` | A re-run gives the same numbers |
| `DAY4_GROUPS`, `DAY4_MONDRIAN` | `docs/day4_groups.json`, `docs/day4_mondrian.json` | Re-runnable records |
| `DAY4_AUDIT`, `DAY4_AUDIT_TABLE` | `docs/day4_audit.json`, `docs/day4_audit.csv` | The one audit; the JSON is the lock |

---

## 4. The pipeline, Day 4 additions

| Script | What it does |
|---|---|
| `groups.py` | Which group each delay is in; learns the code tiers from training rows and merges thin ones. Not run directly. |
| `coverage_stats.py` | Day bootstrap, Bonferroni z, Wilson interval, design effect, three self-tests. Not run directly. |
| `mondrian.py` | One `q_hat` per group, four self-tests. Not run directly. |
| `14_define_groups.py` | Fixes the groups from training rows and calibration counts. Never reads a test row. Refuses once the audit exists. |
| `15_mondrian.py` | Global against Mondrian on temporal and 20 random calibration halves. Checks `q_hat` matches Day 3. Never reads a test row. Re-runnable. |
| `16_audit.py` | Coverage per group from Day 3's per-row file. Loads no model. Checks 7,448 rows and 92.1%. Runs once. |
| `17_plot_groups.py` | Draws the figure from the saved audit. Never touches the test set. |

Commits, in order:

| Commit | Contents |
|---|---|
| `f572385` | Design in the log, settings, the seven scripts |
| `c41502f` | `day4_groups.json` and the Step 3 log entry |
| `ad43299` | Steps 4 to 7 in the log, `day4_mondrian.json`, the prediction in the Day 3 journal |
| `dd92fba` | The audit (JSON, CSV), the figure, Steps 8 and 9 in the log |

---

## 5. What went wrong, and what each one taught me

**1. I copied the old version of a script.**
`15_mondrian.py` came from the Day 3 chat's zip (8,120 bytes), not the
fixed one (8,426). The byte check in Step 3 caught it before the commit.
*Lesson: a size check is cheap and catches the wrong file.*

**2. A log entry missed its commit again.**
The Step 4 and 5 entry was saved after `git add`, so `c41502f` holds only
the Step 3 entry. It went in with `ad43299`, after I had seen calibration
coverage by group, and the log says so. No test coverage had been seen.
*Lesson: same as Day 3. Save, then add, then check `git show --stat`.*

**3. The code-tier design did not foresee an empty tier.**
No known code has a training median between 10 and 20 minutes. The
"> 20 min" tier (43 calibration rows, `MUPR1` and `MUPLB` among them) merged
into "5 to 10 min" (1,566 rows), its only neighbour. The long codes became
under 3% of a tier dominated by six-to-ten-minute codes. I kept the rule as
committed and recorded the cost before the audit: Day 1's `MUPR1`
prediction cannot be tested on this data. Unmerged, 43 rows would have
been description-only anyway.
*Lesson: check how the data falls across the cutoffs, not only where the
cutoffs are.*

**4. Numbers from Claude's stand-in data got into my log.**
SHP was described as 232 test delays (about 3%); the real figure is 304
(4.1%). The conclusion did not change. The correction is in the Step 8
entry, not an edit to committed text.
*Lesson: a number in my log needs to come from my output.*

**5. The pre-run check found one real fix.**
Before I ran anything, the scripts were run on a stand-in with my schema.
`15_mondrian.py` crashed if the model returned pandas Series instead of
arrays, because the calibration halves are picked by position and the
time-sorted slice's labels are not 0 to n − 1. Fixed by converting to
arrays. I also printed the JSON keys and per-row dtypes the scripts read.
*Lesson: check what a script reads, not only what it computes.*

**6. I deferred the hand cases.**
Assignment deadlines. The self-tests passed, including case 3 in code. The
cases and their answers are in section 9 for review before the interview.

---

## 6. Results

### Groups (`14_define_groups.py`)

Line and hour-band calibration counts match `07_verify.py` exactly.

| Code tier | Codes | Training rows | Calibration rows | Medians | Longest |
|---|---|---|---|---|---|
| median <= 5 min | 45 | 14,350 | 4,592 | 3 to 5 | `EUBK`, `EUDO`, `EUCD`, `EUNT` |
| median > 5 min | 19 | 4,804 | 1,609 | 6 to 70 | `MUPR1`, `MUPLB`, `SUUT`, `SUEAS` |
| rare or new code | — | 650 | 362 | — | — |

### Mondrian on calibration (`15_mondrian.py`)

Random halves, global against Mondrian:

| Group | Global | Mondrian | Median bound |
|---|---|---|---|
| median > 5 min | 87.6% | 89.9% | 20.2 → 23.9 min |
| overnight 00–04 | 88.4% | 90.3% | 13.5 → 15.4 min |
| early 05–06 | 87.8% | 90.1% | 10.9 → 12.8 min |
| rare or new code | 92.1% | 90.3% | 19.4 → 15.6 min |
| AM peak 07–09 | 92.4% | 90.2% | 10.7 → 9.7 min |
| SHP | 91.8% | 90.4% | 11.0 → 9.0 min |

Temporal halves: Mondrian widened the spread across groups in every
grouping (line 4.5 → 11.2 points, hour band 2.5 → 5.3, code tier
1.6 → 6.4). SHP fell from 87.6% to 81.4%.

### The audit (`16_audit.py`, once, 7,448 test delays)

| Grouping | Group | n | Coverage | 95% CI | Adjusted CI | deff |
|---|---|---|---|---|---|---|
| line | YU | 3,977 | 93.1% | 92.2–93.9 | 91.8–94.3 | 1.2 |
| line | BD | 3,167 | 90.7% | 89.5–91.9 | 88.9–92.5 | 1.5 |
| line | SHP | 304 | 93.1% | 90.3–95.9 | 89.0–97.2 | 1.0 |
| hour band | overnight 00–04 | 505 | 89.9% | 87.0–92.8 | 85.6–94.2 | 1.2 |
| hour band | early 05–06 | 646 | 91.3% | 88.7–94.0 | 87.4–95.2 | 1.5 |
| hour band | AM peak 07–09 | 1,194 | 93.3% | 91.8–94.8 | 91.1–95.5 | 1.1 |
| hour band | midday 10–15 | 2,087 | 93.1% | 91.9–94.3 | 91.3–94.9 | 1.2 |
| hour band | PM peak 16–18 | 1,347 | 91.0% | 89.3–92.7 | 88.5–93.5 | 1.3 |
| hour band | evening 19–23 | 1,669 | 91.7% | 90.2–93.1 | 89.5–93.8 | 1.2 |
| code tier | median <= 5 min | 5,373 | 92.8% | 92.0–93.5 | 91.7–93.9 | 1.2 |
| code tier | median > 5 min | 1,528 | 89.6% | 87.9–91.3 | 87.0–92.1 | 1.3 |
| code tier | rare or new code | 547 | 92.0% | 88.5–95.4 | 86.9–97.0 | 2.3 |

How to read it:

- **No group undercovers, and none is "possible".**
- **The two low groups repeat.** "> 5 min" and overnight were low on
  calibration (87.6%, 88.4%) and are low on test (89.6%, 89.9%). A pattern
  across both years, not a finding at this power.
- **BD gained less from the 2026 shift than YU** (90.7% against 93.1%),
  though they were level on calibration. Described, not explained.
- **Power.** The adjusted intervals run about ±1 to ±3 points in the large
  groups and ±4 to ±5 in the small ones. A large group would need coverage
  below roughly 87 to 89% to be flagged.
- **Rare or new codes cluster by day** (design effect 2.3) and are a bigger
  share of test (7.3%) than of calibration (5.5%): 2026 brought more codes
  the model never learned.
- **Spread:** line 2.4 points, hour band 3.4, code tier 3.2.

### The figure

`docs/figures/day4_group_coverage.png`: one panel per grouping, a dot for
coverage, a thick bar for the 95% interval, a thin bar for the adjusted
interval. Every thin bar crosses the 90% line, matching the table.

### My predictions against the results

| Predicted | Got | |
|---|---|---|
| No group flagged | none flagged, none "possible" | right |
| Lowest group: > 5 min, about 90% (88.5–91.5) | 89.6% | right |
| Lowest hour bands: early and overnight, 90–91% | overnight 89.9% lowest; then PM peak 91.0%, early 91.3% | half right |
| Rare or new code above overall, about 93% | 92.0%, level with overall | wrong, narrowly |
| Day 1: hour band a weak grouping | largest spread of the three | not supported (the merge diluted code tier) |
| Day 2: rare or new code undercovers first | 92.0% | not supported |
| Day 1: `MUPR1` undercovers | merged away | not testable |

---

## 7. Decisions, each with its reason

| Decision | Reason |
|---|---|
| Mondrian on calibration only, never scored on test | Choosing it after seeing test groups would tune to the test set |
| Code tier by training median, cut at 5, 10, 20 | Tests the Day 1 prediction directly; training rows cannot leak coverage |
| Thin tiers merge; rare or new code never merges | Contiguous ranges; rare codes have no neighbour on that scale |
| Kept the merge that swallowed the long tier | The rule was fixed in advance, and 43 rows could not have been flagged anyway |
| Day bootstrap, normal interval around its SE | Delays on one day are linked; the 0.2% tails need the normal curve, not a few thousand draws |
| Bonferroni over 12 groups | Without it, about a 46% chance of a false flag |
| Prediction committed before the audit (`ad43299`, 16:29) | The commit time proves the order |
| Audit runs once; groups locked after it | No slicing until something breaks |
| Nothing changed after seeing the audit | Rule 2 |

---

## 8. What Day 5 has to say

- The README's first three lines: what it does, 92.1% marginal coverage
  against 90%, and that no line, hour band or code tier undercovers.
- Both figures, and the caveat for each: drift for Day 3, power for Day 4.
- Mondrian as what I would use if a group did undercover, with its cost in
  minutes, and why it does not fix drift.
- Adaptive conformal inference as the next step, for drift.
- The limits, plainly: retrospective (needs the delay code), marginal
  guarantee, long codes too rare to test.

---

## 9. Hand cases (deferred; answers for review)

**1. Is 85% evidence of undercovering?**
340 / 400 = 0.85. ± 1.96 × √(0.85 × 0.15 / 400) = ± 1.96 × 0.0179 = ± 3.5
points, so 81.5% to 88.5%: entirely below 90%, so evidence. With 34 / 40,
± 1.96 × √(0.85 × 0.15 / 40) = ± 11.1 points, so 73.9% to 96.1%: includes
90%, so not evidence. Same coverage, ten times fewer rows.

**2. Why Bonferroni?**
Fourteen groups at 95%, none truly undercovering: 1 − 0.95¹⁴ = 1 − 0.488 =
51% chance of at least one false flag. Testing each at 0.05 / 14 = 0.0036
keeps it at 5%. For my 12 groups: 46%, and 0.05 / 12 = 0.0042 (z = 2.87).

**3. Mondrian at 50%.**
Scores A: 1, 2, 3; B: 10, 20, 30. Global: k = ceil(7 × 0.5) = 4, the 4th
smallest of 1, 2, 3, 10, 20, 30 is 10. Every new A score is covered; a new
B score is covered only if it lands in the lowest of B's 4 positions,
about 1 in 4. Per group: k = ceil(4 × 0.5) = 2, so A gets 2 and B gets 20,
and each covers 2 of 4 positions, 50%: each group gets the promise.

**4. Why resample days?**
A signal failure makes 12 long delays on one afternoon and the bound misses
all of them. Counted as 12 independent misses, they look like 12 pieces of
evidence; really they are close to one event. Treating linked rows as
independent overstates the information and makes the interval too narrow.
The design effect measures by how much.

---

## 10. Questions I should be able to answer quickly

**Overall coverage was 92.1%. Why audit groups at all?**
It is a weighted average. A small group can sit at 80% while the rest hold
the average up; with SHP at 4% of delays, the overall figure would barely
move.

**What did the audit find?**
No line, hour band or code tier undercovers 90% under a Bonferroni rule
fixed in advance. The lowest are the longer-code tier (89.6%) and overnight
(89.9%), both with 95% intervals that include 90%.

**So conditional coverage holds?**
Not shown to fail in these groups at this power. The audit rules out large
undercoverage, not a point or two, and it says nothing about groups I did
not define, such as single codes or stations.

**What happened to your MUPR1 prediction?**
No known code has a median between 10 and 20 minutes, so the long-code tier
(43 calibration rows) merged into the 5 to 10 tier. I kept the rule I had
fixed and recorded, before the audit, that the long-code question is out of
reach on this data.

**Why not apply Mondrian to the test set?**
Choosing it after seeing which test groups miss, then scoring it on test,
tunes the method to the test set. A clean test needs new months.

**Would Mondrian have helped?**
On random calibration halves, yes: it lifted the three low groups to about
90%, at 1.9 to 3.7 extra minutes of median bound. On temporal halves it
made things worse, because it does not correct for drift.

**Why resample days, not rows?**
Delays on one bad day are linked. Treating them as independent makes the
interval too narrow; the design effect measures by how much.

**Why Bonferroni, and what does it cost?**
Twelve groups at 95% give about a 46% chance of a false flag. Testing each
at 0.05 / 12 keeps it at 5%; the cost is power.

**What does a design effect of 2.3 mean?**
The rare-or-new-code group carries about the information of 547 / 2.3 ≈ 240
independent delays: its misses come in clusters on particular days.

**Why is BD lower than YU?**
I do not know. They were level on calibration; BD gained less from the 2026
shift. I report it as an observation.

---

## 11. Commands I run to check everything

```powershell
python src/07_verify.py           # expect "All 31 checks passed"
python src/14_define_groups.py    # refuses: the audit has run
python src/15_mondrian.py         # calibration only; global q_hat 0.0835; safe to re-run
python src/16_audit.py            # prints the stored audit and refuses to run again
python src/17_plot_groups.py      # redraws the figure from the saved audit
```

---

## 12. Open items

- [ ] Write the AI acknowledgement below, and the Day 4 AI note in `log.md`
- [ ] Work the four hand cases on paper before the interview (section 9)
- [ ] `log.md` Day 3 Step 7: "covered 90% of 2025 calibration delays"
      should say "at least 90.0%" (if not already fixed)
- [ ] Day 5: README with both figures, write-up, `requirements.txt` in a
      fresh environment, repo public
- [ ] Day 5, after the write-up: decide on one static interactive page

---

## AI acknowledgement

Claude wrote all seven Day 4 scripts, including the `15_mondrian.py` array fix, and drafted this journal, the log entries and the audit prediction from my output; two numbers from its stand-in data (SHP 232, about 3%) reached my log and are corrected in the Step 8 entry. I ran every step, caught the outdated `15_mondrian.py` with a byte check, and committed the audit design before any group was looked at.
