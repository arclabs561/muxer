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

## Applied harness examples

These are intentionally domain-flavored, offline loops that mirror how external systems
use `muxer` for matrix routing (slice selection + backend selection + feedback):

- `cargo run --example matrix_harness`
  - generic eval matrix (`task x dataset x backend`) with coverage + guardrails.
- `cargo run --example pcap_triage_harness`
  - network-security/PCAP triage (`dataset x protocol x environment`) with protocol compatibility.
- `cargo run --example synthetic_drift_harness`
  - controlled drift world where one backend regresses on hard slices after a known epoch.
- `cargo run --example ad_auction_harness`
  - recommender/ad-auction traffic cells (`objective x geo x device`) with post-privacy-shift drift.

## How to read the output

- **`det_rate` / `ok`**: fraction of trials that alarmed after the change within the horizon.
- **`fa_rate` / `fa`**: fraction of trials that alarmed before the change.
- **`wall(mean/p90)`**: wall-clock delay after the change, mean / 90th percentile (over detected trials).
- **`post(mean/p90)`**: post-change samples from the changed arm, mean / p90 (over detected trials).
- **`wrong`** (when present): alarmed after the change, but on the wrong arm.

## Concepts (minimal)

### Two clocks

In bandit sensing, you only observe an arm when you sample it.

- **Wall time** \(t\): global decision steps.
- **Sample time** \(n_k\): observations from a particular arm.

A useful approximation is \(delay\_{wall} \approx delay\_{samples} / sampling\_{rate}\).
Coverage lets you set a **sampling rate floor**.

### Drift feature vs alarm

- `CatKlDetector` (cumulative drift): good drift *feature*, can have inertia (pre-history matters).
- `CusumCatDetector` (reflected LLR): designed for low post-change sample delay; good *alarm*.

### False alarms (the contract)

`detector_calibration` and `bqcd_calibrated` use: \(P_\infty[\tau < m] \le \alpha\).
Interpretation: under the null, you should almost always **survive at least** \(m\) wall steps.

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
- Calibrates thresholds under the null to satisfy \(P_\infty[\tau < m] \le \alpha\) at a given sampling interval.
- Evaluates post-change sample delays under a fixed shift.
- Uses the same “threshold-free max-score” trick as `bqcd_calibrated`:
  simulate the null once per trial with `threshold=+∞`, record \(M=\max_{t<m} S(t)\), then pick the smallest
  grid threshold \(h\) such that \(\hat P[M \ge h] \le \alpha\) (optionally requiring a Wilson upper bound).

Notes:
- You can control the calibration conservatism via `CAL_Z` (default `1.96`) and `CAL_REQUIRE_WILSON=1`.

Takeaway:
- To enforce a strict “survive to \(m\)” constraint, CUSUM thresholds can need to be **much larger**
  when sampling is frequent.
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

### 6) `bqcd_calibrated.rs`

Command:

```bash
cargo run --release --example bqcd_calibrated --features stochastic
```

What it does:
- Same “exactly one arm changes at time \(\nu\)” world as `bqcd_sampling`, but adds **two knobs**:
  - **Global multi-arm false-alarm calibration**: for each (policy, alt-bank) pair, choose a threshold \(h\)
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

