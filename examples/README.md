# muxer examples

Each example answers one routing question and is runnable from the repo root.
Output excerpts below are real, captured from local runs. For derivations,
failure-mode discussion, and longer experiment notes, see
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Start Here

### `validation_trace_matrix`: how do the generic and quality paths behave on real data?

Downloads are kept outside the repository. The script builds one common trace
format from categorical, numeric, mixed, and missing-value UCI datasets. The
example replays the traces through the quality-oriented `Router` and, separately,
through `select_candidate_assessments` using rolling utility, cost, and latency
vectors. The generic path is a full-information offline replay; the Router path
only observes its selected arm.

```bash
scripts/fetch_validation_datasets.sh data/uci
uv run scripts/build_validation_traces.py --input data/uci --output data/traces/classification-traces.csv
cargo run --example validation_trace_matrix -- data/traces/classification-traces.csv
```

The trace fields are deliberately explicit (`label`, `predicted`, `cost_units`,
`elapsed_ms`). A caller with a different schema can use
`select_candidate_assessments` directly instead of manufacturing `Outcome`
flags.

The current local trace has 6,499 mushroom, 1,382 car, 36,168 bank, 1,279 red
wine, 3,918 white wine, and 39,073 adult evaluation rows. The resulting
quality-router and generic-selector accuracies are printed per dataset; the
per-row oracle is an offline upper reference, not a deployment estimate.

### `getting_started`: what does the basic routing loop do?

Creates three backends, lets the router explore each one, records finalized outcomes,
and shows how `quality_weight` resolves a quality-versus-cost tradeoff.

```bash
cargo run --example getting_started
```
```text
=== First 6 rounds (explore-first) ===
  round  0: chose gpt-4o           quality=0.92  junk=false  prechosen=["gpt-4o"]
  round  1: chose claude-sonnet    quality=0.78  junk=false  prechosen=["claude-sonnet"]
  round  2: chose gemini-pro       quality=0.55  junk=true  prechosen=["gemini-pro"]

=== After 30 rounds ===
  gpt-4o           calls=10  ok=1.00  junk=0.00  quality=0.92
  claude-sonnet    calls=10  ok=1.00  junk=0.00  quality=0.78
  gemini-pro       calls=10  ok=1.00  junk=1.00  quality=0.55

Best arm now: "claude-sonnet"  (mode: Normal)

With quality_weight=1.0: best arm is "gpt-4o"
```

### `guardrail_semantics`: where does latency filtering run?

Compares the fallback pipeline, where novelty and coverage can pre-pick arms
before latency filtering, with the filter-first pipeline, where the empirical
filter is strict inside the novelty/coverage/selector policy stage.

```bash
cargo run --example guardrail_semantics
```
```text
== guardrail semantics demo ==
-- novelty + require_measured --
fallback (novelty first): chosen=["unseen"]
filter-first: chosen=["seen"], stopped_early=false

-- coverage + require_measured --
fallback (coverage first): chosen=["c"]
filter-first: chosen=["b"], stopped_early=false
```

## Routing Harnesses

### `matrix_harness`: does coverage touch every slice?

Runs an offline matrix over task, dataset, and backend cells. This is the
smallest applied harness.

```bash
cargo run --example matrix_harness
```
```text
== matrix_harness summary ==
ner.lang=en.dom=news         calls= 36
ner.lang=en.dom=social_media calls= 36
ner.lang=en.dom=biomedical   calls= 36
re.lang=en.dom=wikipedia     calls= 36
coref.lang=en.dom=news       calls= 36
```

### `llm_gateway_harness`: what happens after a model regresses?

Simulates model-gateway traffic with quality, latency, cost, and a prompt-change
drift point.

```bash
cargo run --release --example llm_gateway_harness
```
```text
== llm_gateway_harness ==
480 deterministic requests, drift at round 240: balanced degrades on extraction.

pre-drift aggregate
model           calls  ok     junk   quality  cost/call  ms/call
balanced          108  1.000  0.139    0.831       4.00    278.0
verifier           96  0.990  0.062    0.904      12.00    643.3

post-drift aggregate
model           calls  ok     junk   quality  cost/call  ms/call
balanced           36  0.972  1.000    0.650       4.00    338.9
local-small        57  0.965  0.140    0.796       1.00    137.5
verifier          137  1.000  0.066    0.889      12.00    631.5
```

### `uci_mushroom_router`: does routing work on a real categorical dataset?

Replays the UCI Mushroom data through three fixed policies: a majority-class
baseline, an odor-only classifier, and a categorical naive-Bayes classifier.
The router sees only the selected policy's outcome; the full trace is used as
an offline per-row oracle reference.

```bash
curl -L https://archive.ics.uci.edu/static/public/73/mushroom.zip -o /tmp/mushroom.zip
unzip -o /tmp/mushroom.zip -d /tmp/mushroom
cargo run --example uci_mushroom_router -- /tmp/mushroom/agaricus-lepiota.data
```

The example exits successfully with a clear message when no dataset path is
provided, so the default build does not depend on a download.

### `synthetic_drift_harness`: which backend changes after drift?

Uses a controlled drift world where hard slices regress after a known epoch.

```bash
cargo run --example synthetic_drift_harness
```
```text
== synthetic_drift_harness ==
slice coverage:
  family=vision   hardness=easy region=us calls= 36
  family=ranking  hardness=hard region=us calls= 34
hard-slice backend picks pre/post drift:
  balanced_nn        pre= 25 post= 21
  hard_specialist    pre= 18 post= 16
  large_transformer  pre= 17 post=  7
  tiny_rule          pre= 12 post= 20
```

### `contextual_router`: how does LinUCB use context?

Requires the `contextual` feature. The example prints periodic decisions and
ends with the learned split on high-difficulty contexts.

```bash
cargo run --example contextual_router --features contextual
```
```text
t=   0 ctx=[0.17325464426155657, 0.15229643060221798] decision=Decision { policy: LinUcb, chosen: "small", probs: None, notes: [ExploreFirst] } reward=0
t= 200 ctx=[0.5287534261595772, 0.7204979174860174] decision=Decision { policy: LinUcb, chosen: "big", probs: None, notes: [DeterministicChoice] } reward=0
t=1800 ctx=[0.8879784495223955, 0.7420458346578667] decision=Decision { policy: LinUcb, chosen: "big", probs: None, notes: [DeterministicChoice] } reward=1
high-difficulty picks (t>=1000): big=204 small=0
```

### `off_policy_evaluation`: how do propensities change offline estimates?

Shows a logged sample where the naive mean is biased by the logging policy and
IPS corrects it for a target policy.

```bash
cargo run --example off_policy_evaluation
```
```text
naive mean: 0.800
IPS estimate: 0.500
self-normalized IPS estimate: 0.500
```

## Drift Experiments

### `significant_shift_sim`: should every CUSUM alarm restart routing?

Keeps the best arm unchanged while a worse arm degrades. Restarting on every
CUSUM alarm repeatedly relearns a harmless shift; a best-arm-aware gate
suppresses those restarts.

```bash
cargo run --example significant_shift_sim --features stochastic
```
```text
strategy                 | restarts alarms suppressed pulls_drifted mean_reward regret
restart_on_any_cusum     |    11.50  11.50       0.00        1062.0      0.8731   285.0
significant_gate         |     0.14   8.88       8.73         854.9      0.8882   224.1
```

### `coverage_autotune`: how does a wall-delay target become coverage?

Requires the `stochastic` feature. It converts a target wall delay into a
coverage floor and checks the empirical detection table.

```bash
cargo run --example coverage_autotune --features stochastic
```
```text
== coverage_autotune ==
K=6 nu=20000 horizon=60000 trials=250
CUSUM: alpha=0.001 min_n=20 thr=12.0 | KL(p1||p0)≈0.045241 => pred_samples h/KL≈265.2
targetW | cov_frac  pred_wall | fa   ok  wrong | wall(mean/p90)  post(mean/p90)  post_rate  mean_frac
   1000 |  0.1667*    1591.5 | 0.004 0.996 0.000 |  1498.5/  2326     250.8/   389     0.167     0.167
   5000 |  0.0530     5000.0 | 0.000 1.000 0.000 |  1016.4/  2332     259.0/   399     0.446     0.170
  20000 |  0.0133    20000.0 | 0.000 1.000 0.000 |  2891.1/  6917     254.0/   396     0.349     0.193
```

### `bqcd_sampling`: what is the cost of focusing without coverage?

Requires the `stochastic` feature. The table compares round-robin, focus by
cumulative drift score, focus by CUSUM score, and CUSUM with coverage.

```bash
cargo run --example bqcd_sampling --features stochastic
```
```text
nu=20000 horizon=40000 alarm=CUSUM(min_n=30,thr=12) alt=[0.05, 0.05, 0.45, 0.45]
policy                              | fa_rate det_rate  wall(mean/p90)  post(mean/p90)  mean_frac_on_chg
round_robin                         |  0.025  0.595   3533.0/ 13038     589.4/  2174            0.167
eps_focus(eps=0.020, catkl, cov(off)) |  0.040  0.565   6366.0/ 17193    2430.1/  7741            0.236
eps_focus(eps=0.020, cusum, cov(off)) |  0.040  0.315   2088.8/  8667      24.1/    57            0.161
eps_focus(eps=0.020, cusum, cov(frac=0.020,floor=10)) |  0.025  0.385   2098.7/  8265      52.5/   176            0.148
```

## More

Other runnable harnesses cover PCAP triage, ad auctions, fraud scoring, search
ranking, medical triage, Thompson sampling, EXP3-IX, monitored routing, sticky
MAB behavior, and delayed junk labels. See `cargo run --example <name>` and
[`EXPERIMENTS.md`](EXPERIMENTS.md) for the longer experiment set.
