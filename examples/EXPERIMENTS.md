# `muxer` mini-experiments (bandits × monitoring × drift geometry)

This folder contains small, runnable experiments that probe the interactions between:

- **Sampling policy** (who gets pulled, how often)
- **Detection statistic / alarm rule** (CUSUM vs cumulative drift scores)
- **Time scales** (wall-clock steps vs per-arm sample counts)

The goal is not “a full framework”; it’s to make the **failure modes and tradeoffs inspectable**.

## Key concepts (one page)

### Two clocks: wall time vs sample time

In bandit sensing, you only observe arm \(k\) when you sample it.

- **Wall-clock time**: the global decision index \(t\in\{0,1,2,\dots\}\).
- **Sample time**: the number of observations collected from a particular arm (or after a change).

Detection delay in wall time is roughly:

\[
\text{delay\_wall} \approx \frac{\text{delay\_samples}}{\text{sampling\_rate}}
\]

So “quick detection” is meaningless unless you say **how often** you sample.

### Two detector families

These experiments repeatedly show a stark split:

- **Cumulative drift scores** (e.g. empirical-distribution KL-to-baseline) are great as *features*,
  but can exhibit **inertia** that grows with pre-change history.
- **Resetting / forgetful alarms** (CUSUM-style reflected LLR, windowed tests) are designed to keep
  post-change **sample delay** small and relatively insensitive to pre-history.

In `muxer::monitor`:

- `CatKlDetector`: \(S_n = n \cdot KL(\hat q_n \| p_0)\)
- `CusumCatDetector`: \(S_t = \max(0, S_{t-1} + \log p_1(X_t) - \log p_0(X_t))\)

### False alarms: “per-horizon” vs BQCD-style

There are multiple reasonable constraints; they are not equivalent.

`detector_calibration` uses a BQCD-style version:

- \(P_\infty[\tau < m] \le \alpha\)

Interpretation: under the null (no change), you should almost surely **survive at least** \(m\)
wall-clock steps.

This differs from “alarm rate over an infinite horizon” (where many tests will eventually alarm).

## Experiments

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

Takeaways (typical patterns):
- Focusing on **CatKL score** can be actively harmful: it’s cumulative and can lock onto noise.
- Focusing on a **forgetful suspicion score** (CUSUM) is sample-efficient *when it’s right*, but can
  under-sample the actually-changed arm unless you enforce **coverage**.
- Adding a small coverage floor is often the difference between:
  “fast when lucky” and “reliably fast”.

### 5) `bqcd_calibrated.rs`

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

