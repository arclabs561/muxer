# `muxer` examples: make routing tradeoffs visible

Most programs are small enough to expose one failure mode directly in their
output: starvation, detector inertia, false alarms, and “fast on mean” versus
“reliable on tail.” The real-data matrix is larger because it preserves four
source schemas and their provenance.

## Start here (recommended order)

- `cargo run --example guardrail_semantics`
  - shows where empirical latency filtering runs relative to novelty/coverage.
- `cargo run --example coverage_autotune --features stochastic`
  - turns a wall-delay target into a `CoverageConfig` floor using $h/KL$, then simulates.
- `cargo run --example bqcd_sampling --features stochastic`
  - shows why “focus” without coverage can look great on mean and bad on det_rate/p90.
- `cargo run --example significant_shift_sim --features stochastic`
  - shows why a harmless non-best-arm drift should not necessarily restart routing.
- `cargo run --example off_policy_evaluation`
  - shows why logged rewards need propensities before offline target-policy evaluation.
- `cargo run --example feedback_regime_matrix`
  - applies separate evaluators to logged bandit, algorithm-selection,
    fuzzer-coverage, and annotated time-series data.

## Applied harness examples

These are intentionally domain-flavored, offline loops that mirror how external systems
use `muxer` for matrix routing (slice selection + backend selection + feedback):

- `cargo run --example matrix_harness`
  - generic eval matrix (`task x dataset x backend`) with coverage + guardrails.
- `cargo run --release --example llm_gateway_harness`
  - model-gateway traffic (`task x model`) with quality, latency, cost, and post-prompt-change drift.
- `cargo run --example pcap_triage_harness`
  - network-security/PCAP triage (`dataset x protocol x environment`) with protocol compatibility.
- `cargo run --example synthetic_drift_harness`
  - controlled drift world where one backend regresses on hard slices after a known epoch.
- `cargo run --example ad_auction_harness`
  - recommender/ad-auction traffic cells (`objective x geo x device`) with post-privacy-shift drift.
- `cargo run --example fraud_scoring_harness`
  - payments risk traffic cells (`channel x region x segment`) with post-shift scorer degradation.
- `cargo run --example search_ranking_harness`
  - search traffic cells (`intent x locale x device`) with retrieval-model drift and latency constraints.
- `cargo run --example medical_triage_harness`
  - triage cells (`setting x acuity x cohort`) with protocol-shift degradation in critical cohorts.
- `cargo run --example uci_mushroom_router -- /path/to/agaricus-lepiota.data`
  - real categorical trace from the UCI Mushroom dataset, with fixed classifier policies and an offline per-row oracle.
- `cargo run --example validation_trace_matrix -- data/traces/classification-traces.csv`
  - replays categorical, numeric, mixed, and missing-value UCI traces through both the quality Router and the generic metric-vector selector.

The multi-dataset trace uses UCI Mushroom (categorical), Car Evaluation
(categorical), Bank Marketing (mixed categorical/numeric), Wine Quality
(numeric), and Adult (mixed with missing-value categories). Run
`scripts/fetch_validation_datasets.sh` followed by
`uv run scripts/build_validation_traces.py` to reproduce it. Downloads and
generated CSVs live under the ignored `data/` directory.

## How to read the output

- **`det_rate` / `ok`**: fraction of trials that alarmed after the change within the horizon.
- **`fa_rate` / `fa`**: fraction of trials that alarmed before the change.
- **`wall(mean/p90)`**: wall-clock delay after the change, mean / 90th percentile (over detected trials).
- **`post(mean/p90)`**: post-change samples from the changed arm, mean / p90 (over detected trials).
- **`wrong`** (when present): alarmed after the change, but on the wrong arm.

## Real-data replay

### `feedback_regime_matrix.rs` (pre-registered)

Hypothesis:
- Native-schema replays will expose limitations hidden by the classification
  traces: lower effective support under Thompson-sampling logs than uniform
  logs, a nonzero gap between order-dependent algorithm selection and a
  per-instance hindsight oracle, median-final-coverage versus cross-trial-IQR
  tradeoffs across fuzzers, and larger marginal four-bin Hellinger distance
  inside annotated windows than held-out normal periods for at least three of
  five streams.

The terminology above was tightened after the run without changing the four
recorded directional predictions.

Method:
- Fetch checksum-pinned artifacts from [Open Bandit Dataset], [ASlib],
  [FuzzBench], and [NAB] into `data/feedback/raw/`.
- Build four separate derived schemas under `data/feedback/derived/`; do not
  coerce clicks, runtimes, edge counts, or anomaly windows into `Outcome` flags.
- Replay the schemas through scalar OPE, metric-vector selection, Pareto
  filtering, and marginal four-bin Hellinger comparisons. Report each dataset
  separately. Keep the per-instance oracle and post-hoc best median as offline
  references only.

Data provenance:
- Source revisions, URLs, byte counts, and SHA-256 digests live in
  `scripts/feedback_sources.toml`. The build writes a deterministic provenance
  record beside the ignored derived data. The evaluator verifies all four
  derived byte counts and digests before computing metrics.

Results:
- The hypothesis above was recorded before the first run. The sensitivity run
  was captured from commit `8447c281f2e54d459093c56c5d214c215e64ea41`:

  ```bash
  uv run scripts/fetch_feedback_datasets.py
  uv run scripts/build_feedback_traces.py
  cargo run --example feedback_regime_matrix
  ```

- The four derived files contained 362,662 rows: 20,000 Open Bandit rows,
  5,150 ASlib runs, 303,134 FuzzBench trajectory points, and 34,378 NAB
  observations.
- Open Bandit: the evaluator-defined target is uniform over item IDs 0 through
  39. The uniform logger had effective sample size 4,995.0 and maximum
  importance weight 2.00. The Bernoulli Thompson-sampling logger had effective
  sample size 177.8 and maximum weight 294.12 despite 3,645 of 10,000 rows
  having nonzero target support. Clicks are sparse and no uncertainty interval
  is reported, so these IPS and SNIPS values diagnose overlap rather than
  establish a policy-value conclusion.
- ASlib: the order-dependent full-information selector's gap to the
  per-instance hindsight oracle changed from 1,314.106 to 1,586.944 PAR10
  under ascending versus descending instance order for CSP-Minizinc, and from
  0.023 to 0.027 accuracy for OPENML-WEKA. The fixed aggregate choice beat the
  history-based selection in both scenarios. The replay uses no instance
  features and updates every algorithm after every instance.
- FuzzBench: a zero spread penalty chose the best median coverage on all 20
  benchmarks. At each tested positive weight (0.05, 0.10, 0.25, and 0.50), the
  selector accepted lower median coverage on one benchmark. This is a post-hoc
  final-coverage aggregation, not online trajectory routing; a zero scalar
  weight also leaves the IQR dimension on the Pareto frontier.
- NAB: marginal four-bin Hellinger distance was larger inside annotated windows
  than in held-out normal rows for NYC taxi, traffic speed, and Twitter AAPL,
  but not for ad-exchange CPC or AWS CPU. The count remained 3 of 5 with
  baseline fractions of one fifth, one third, and one half. The comparison
  ignores temporal ordering and does not evaluate NAB anomaly scores.

Interpretation and limits:
- All four directional predictions held. The FuzzBench result affected only one
  benchmark, and the NAB direction reversed on two streams, so neither should
  be generalized to every workload.
- Native schemas composed with existing muxer primitives without coercion into
  `Outcome`. No one-schema baseline was tested, so the run does not establish
  that separate schemas are necessary or superior. It also does not show that
  the quality `Router` is domain-neutral or justify a generic stateful router.
- The runs are offline diagnostics over 14 pinned artifacts from four source
  projects. Hindsight references and post-hoc aggregates use information
  unavailable to an online policy. No result is a deployment guarantee or a
  cross-dataset ranking of algorithms.

[Open Bandit Dataset]: https://github.com/st-tech/zr-obp
[ASlib]: https://github.com/coseal/aslib_data
[FuzzBench]: https://google.github.io/fuzzbench/
[NAB]: https://github.com/numenta/NAB

### `uci_mushroom_router.rs`

The UCI Mushroom dataset contains 8,124 rows with 22 categorical features and
an edible/poisonous label. The example uses every fifth row to fit the policy
surrogates, then replays the other rows through `Router`. It reports both the
router's selected accuracy and each policy's accuracy over the full evaluation
trace, so exposure bias is visible.

Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/73/mushro),
then run:

```bash
curl -L https://archive.ics.uci.edu/static/public/73/mushroom.zip -o /tmp/mushroom.zip
unzip -o /tmp/mushroom.zip -d /tmp/mushroom
cargo run --example uci_mushroom_router -- /tmp/mushroom/agaricus-lepiota.data
```

The local run on the 8,124-row file produced 6,499 evaluation rows, 98.42%
selected accuracy, 0.54 percentage points of gap to the per-row policy oracle,
and 94 hard failures. These are benchmark observations, not a claim
about deployment performance.

## Concepts (minimal)

### Two clocks

In bandit sensing, you only observe an arm when you sample it.

- **Wall time** $t$: global decision steps.
- **Sample time** $n_k$: observations from a particular arm.

A useful approximation is $delay\_{wall} \approx delay\_{samples} / sampling\_{rate}$.
Coverage targets a minimum empirical sampling share in these simulations.

### Drift feature vs alarm

- `CatKlDetector` (cumulative drift): good drift *feature*, can have inertia (pre-history matters).
- `CusumCatDetector` (reflected LLR): designed for low post-change sample delay; good *alarm*.

### False alarms (simulation target)

`detector_calibration` and `bqcd_calibrated` use: $P_\infty[\tau < m] \le \alpha$.
This target is evaluated in each program's finite synthetic world. It is not a
system-wide guarantee for `Router`.

## Experiments

### 0) `guardrail_semantics.rs`

Command:

```bash
cargo run --example guardrail_semantics
```

What it does:
- Prints two tiny scenarios that show the difference between:
  - the default fallback pipeline (novelty/coverage pre-picks happen before latency filtering), and
  - the filter-first pipeline (latency filtering happens first with no fallback).

Takeaway:
- Choose whether an empirical latency filter runs before or after
  novelty/coverage inside the policy stage. Authoritative safety and readiness
  eligibility belongs in the caller-provided candidate set.

### 1) `free_lunch_investigation.rs`

Command:

```bash
cargo run --example free_lunch_investigation --features stochastic
```

What it does:
- Two arms, one changes at a known time.
- Compare `select_mab_decide`, `ThompsonSampling` (with/without decay), and deterministic selection
  with explicit `CoverageConfig`.
- Reports an empirical Pareto frontier using `pare::ParetoFrontier`:
  reward vs detection delay vs a crude estimation error.
- Detection delay cells are shown as `mean/p90@rate` (mean + 90th percentile over *detected* trials,
  plus the detection rate within the horizon).

Takeaway:
- “Free lunch” depends on whether the arm that changes is **sampled anyway**.
- When an arm is starved, you need either **maintenance sampling** or a policy that forgets (decay).

### 2) `detector_inertia.rs`

Command:

```bash
cargo run --example detector_inertia --features stochastic
```

What it does:
- One categorical stream with a change; you only observe it every `interval` steps.
- Sweeps:
  - sampling interval (rate)
  - pre-change history length
- Compares `CatKlDetector` vs `CusumCatDetector`.

Takeaway:
- CUSUM’s post-change sample delay stays ~constant across many regimes.
- CatKL’s post-change sample delay can grow substantially with pre-history (inertia).

### 3) `detector_calibration.rs`

Command:

```bash
cargo run --example detector_calibration --features stochastic
```

What it does:
- Selects thresholds whose in-sample null alarm estimate meets
  $\hat P_\infty[\tau < m] \le \alpha$ at a given sampling interval.
- Evaluates post-change sample delays under a fixed shift.
- Uses the same “threshold-free max-score” trick as `bqcd_calibrated`:
  simulate the null once per trial with `threshold=+∞`, record $M=\max_{t<m} S(t)$, then pick the smallest
  grid threshold $h$ such that $\hat P[M \ge h] \le \alpha$ (optionally requiring a Wilson upper bound).

Notes:
- You can control the calibration conservatism via `CAL_Z` (default `1.96`) and `CAL_REQUIRE_WILSON=1`.
- The threshold and pointwise Wilson bound use the same trials. Validate the
  selected threshold on independent trials before making an out-of-sample claim.

Takeaway:
- In these simulations, lower in-sample alarm targets require larger CUSUM
  thresholds when sampling is frequent.
- Even after calibration, CUSUM tends to require far fewer post-change samples than cumulative drift scores.

### 4) `bqcd_sampling.rs`

Command:

```bash
cargo run --example bqcd_sampling --features stochastic
```

What it does:
- K-armed bandit sensing: exactly one arm changes at time $\nu$.
- Stops on a CUSUM alarm; compares sampling policies:
  - `round_robin`
  - epsilon-focus on “most suspicious” arm, where suspicion is CatKL vs CUSUM score
  - epsilon-focus with **explicit coverage quotas** (`CoverageConfig`)
- Reports wall/post-change delays as `mean/p90` over detected trials (with `det_rate` shown separately).

Takeaways (typical patterns):
- Focusing on **CatKL score** can be actively harmful: it’s cumulative and can lock onto noise.
- Focusing on a **forgetful suspicion score** (CUSUM) is sample-efficient *when it’s right*, but can
  under-sample the actually-changed arm unless you enforce **coverage**.
- Adding a small coverage floor is often the difference between:
  “fast when lucky” and “reliably fast”.

### 5) `coverage_autotune.rs`

Command:

```bash
cargo run --example coverage_autotune --features stochastic
```

What it does:
- K-armed bandit sensing with one changed arm at time $\nu$, like `bqcd_sampling`.
- Sampling policy: coverage-quota pre-picks, otherwise focus on max CUSUM score.
- “Autotunes” a coverage floor for a target wall delay $W$ using the information-theoretic scaling:
  - predicted post-change samples $N \approx h / KL(p_1\|p_0)$
  - target per-arm sampling floor $r := N / W$ (capped at $1/K$ for feasibility)
- Prints predicted vs empirical behavior, including a `wrong` rate for “alarmed on the wrong arm after $\nu$”.

Takeaway:
- $h/KL$ is a decent predictor of **post-change samples** (the “sample clock”).
- Meeting a **wall-clock** target requires you to translate $h/KL$ into a minimum sampling rate,
  and `CoverageConfig` gives you a direct control for that.

### 6) `significant_shift_sim.rs`

Command:

```bash
cargo run --example significant_shift_sim --features stochastic
```

What it does:
- Two arms, where arm 0 remains best before and after the change.
- Arm 1 degrades at a known time and triggers per-arm CUSUM alarms under
  maintenance sampling.
- Compares:
  - restarting globally on any CUSUM alarm, and
  - suppressing global restart when the alarmed arm is not the estimated best arm.

Takeaway:
- A detector alarm and a route-changing event are different decisions.
- Significant-shift-aware monitoring should first prove that the best arm or
  route-quality ordering changed; otherwise it can spend quality relearning a
  harmless non-best-arm shift.

### 7) `bqcd_calibrated.rs`

Command:

```bash
cargo run --release --example bqcd_calibrated --features stochastic
```

What it does:
- Same “exactly one arm changes at time $\nu$” world as `bqcd_sampling`, but adds **two knobs**:
  - **Toy-world multi-arm false-alarm calibration**: for each (policy, alt-bank) pair, choose a threshold $h$
    so that under the null $P_\infty^\pi[\tau < m] \le \alpha$, where $\tau$ is the first alarm across *any* arm.
  - **Unknown post-change robustification (GLR-lite)**: instead of one $p_1$, run a small bank of CUSUMs
    (one per alternative) and alarm if $\max_j S^{(j)}_t \ge h$.
- Calibration is implemented via a threshold-free statistic:
  - simulate the null once per trial (with “never alarm” threshold),
  - record $M=\max_{t<m,\;arm,\;alt} S_{arm,alt}(t)$ (only after `min_n` for that arm),
  - then pick the smallest grid threshold $h$ with $\hat P[M\ge h]\le \alpha$.
- Reports for each configuration:
  threshold found, estimated null FA rate, shift FA rate, detection rate, mean wall delay,
  mean post-change samples on the changed arm, fraction of pulls spent on the changed arm,
  plus a timing breakdown (`cal` vs `eval`).
- In the current output, the wall/post-delay columns are `mean/p90` over detected trials (with `det_rate`
  shown separately).

Notes:
- Alt banks currently include `alt=single_conservative`, `alt=trio`, and `alt=quad`.
- You can override Monte Carlo counts via env vars: `CAL_TRIALS` and `EVAL_TRIALS`.
- Calibration reports a Wilson upper bound `hi` (set `CAL_Z`, default `1.96`). If you want the
  calibration to *require* `hi <= alpha`, set `CAL_REQUIRE_WILSON=1` (you’ll usually need a much larger `CAL_TRIALS`).
- The separate evaluation trials test the selected threshold only within this
  synthetic policy, arm set, horizon, and reset model.

Takeaways (from a representative run; Monte Carlo noise is real):
- `cusum_max` focusing can yield **tiny post-change sample counts** (tens of samples) but can also yield
  a **lower detection rate** because it sometimes starves the changed arm; adding coverage tends to raise
  `frac_on_changed` and detection rate, at some cost in wall delay.
- `catkl` focusing is a useful negative control: it tends to spend **a lot** of samples on the changed arm
  (high `frac_on_changed`) but still can have **large wall delays** due to CatKL’s inertia + mis-focus risk.
- The `trio` alt-bank can materially change outcomes vs `single_conservative` (sometimes improving wall delay,
  sometimes changing shift false alarms), which is exactly why “unknown post-change” is worth modeling explicitly.

## External reading that shaped these experiments

- **Bandit Quickest Changepoint Detection** (Gopalan, Saligrama, Lakshminarayanan), `arXiv:2107.10492`
  - `https://arxiv.org/abs/2107.10492`
- CUSUM background (Page’s scheme / sequential analysis):
  - `https://en.wikipedia.org/wiki/CUSUM`
