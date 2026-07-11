# `muxer` examples: make routing tradeoffs visible

These programs are intentionally small. They exist so you can **see** the failure modes in the output:
starvation, detector inertia, false alarms, and “fast on mean” vs “reliable on tail”.

## Start here (recommended order)

- `cargo run --example guardrail_semantics`
  - shows what “soft” vs “guardrail-first (strict)” means when novelty/coverage is enabled.
- `cargo run --example coverage_autotune --features stochastic`
  - turns a wall-delay target into a `CoverageConfig` floor using \(h/KL\), then simulates.
- `cargo run --example bqcd_sampling --features stochastic`
  - shows why “focus” without coverage can look great on mean and bad on det_rate/p90.
- `cargo run --example significant_shift_sim --features stochastic`
  - shows why a harmless non-best-arm drift should not necessarily restart routing.
- `cargo run --example off_policy_evaluation`
  - shows why logged rewards need propensities before offline target-policy evaluation.

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

- **Wall time** \(t\): global decision steps.
- **Sample time** \(n_k\): observations from a particular arm.

A useful approximation is \(delay\_{wall} \approx delay\_{samples} / sampling\_{rate}\).
Coverage targets a minimum empirical sampling share in these simulations.

### Drift feature vs alarm

- `CatKlDetector` (cumulative drift): good drift *feature*, can have inertia (pre-history matters).
- `CusumCatDetector` (reflected LLR): designed for low post-change sample delay; good *alarm*.

### False alarms (simulation target)

`detector_calibration` and `bqcd_calibrated` use: \(P_\infty[\tau < m] \le \alpha\).
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
  - the default “soft” pipeline (novelty/coverage pre-picks happen before guardrails), and
  - the “guardrail-first” pipeline (strict guardrails happen first).

Takeaway:
- Decide whether guardrails are **soft preferences** or **hard constraints**, and pick the pipeline accordingly.

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
  \(\hat P_\infty[\tau < m] \le \alpha\) at a given sampling interval.
- Evaluates post-change sample delays under a fixed shift.
- Uses the same “threshold-free max-score” trick as `bqcd_calibrated`:
  simulate the null once per trial with `threshold=+∞`, record \(M=\max_{t<m} S(t)\), then pick the smallest
  grid threshold \(h\) such that \(\hat P[M \ge h] \le \alpha\) (optionally requiring a Wilson upper bound).

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
- K-armed bandit sensing: exactly one arm changes at time \(\nu\).
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
- K-armed bandit sensing with one changed arm at time \(\nu\), like `bqcd_sampling`.
- Sampling policy: coverage-quota pre-picks, otherwise focus on max CUSUM score.
- “Autotunes” a coverage floor for a target wall delay \(W\) using the information-theoretic scaling:
  - predicted post-change samples \(N \approx h / KL(p_1\|p_0)\)
  - target per-arm sampling floor \(r := N / W\) (capped at \(1/K\) for feasibility)
- Prints predicted vs empirical behavior, including a `wrong` rate for “alarmed on the wrong arm after \(\nu\)”.

Takeaway:
- \(h/KL\) is a decent predictor of **post-change samples** (the “sample clock”).
- Meeting a **wall-clock** target requires you to translate \(h/KL\) into a minimum sampling rate,
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
- Same “exactly one arm changes at time \(\nu\)” world as `bqcd_sampling`, but adds **two knobs**:
  - **Toy-world multi-arm false-alarm calibration**: for each (policy, alt-bank) pair, choose a threshold \(h\)
    so that under the null \(P_\infty^\pi[\tau < m] \le \alpha\), where \(\tau\) is the first alarm across *any* arm.
  - **Unknown post-change robustification (GLR-lite)**: instead of one \(p_1\), run a small bank of CUSUMs
    (one per alternative) and alarm if \(\max_j S^{(j)}_t \ge h\).
- Calibration is implemented via a threshold-free statistic:
  - simulate the null once per trial (with “never alarm” threshold),
  - record \(M=\max_{t<m,\;arm,\;alt} S_{arm,alt}(t)\) (only after `min_n` for that arm),
  - then pick the smallest grid threshold \(h\) with \(\hat P[M\ge h]\le \alpha\).
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
